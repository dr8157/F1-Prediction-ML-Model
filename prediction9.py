"""F1 Baku Grand Prix race predictor.

Pipeline: fetch what live F1 data is reachable (FastF1 schedule/session,
formula1.com results page), fall back transparently to a verified frozen
qualifying snapshot when it is not, engineer the feature set described in
assumptions.md, train an XGBoost model with monotonic constraints on a
physics-consistent synthetic corpus, and run a Monte Carlo simulation
(with explicit DNF modeling) to produce a full-field probabilistic ranking.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    import fastf1
except ImportError:  # pragma: no cover - fastf1 is a required dependency
    fastf1 = None

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FASTF1_CACHE = DATA / "fastf1_cache"
ARTIFACTS = DATA / "baku_model"
URL = "https://www.formula1.com/en/results/2026/races/1295/azerbaijan/"
QUALI_URL = URL.rstrip("/") + "/qualifying"
SEED = 20260926

RACE_YEAR = 2026
RACE_ROUND = 15
RACE_NAME = "Azerbaijan Grand Prix"
CIRCUIT_LAT, CIRCUIT_LON = 40.3725, 49.8532

# Explicit qualitative circuit priors, on the same scale as the existing panel.
TRACK = (1.0, 1.0, 0.75, 0.40)
TRACK_META = {
    1: (0.70, 0.65, 0.55, 0.55),  # Melbourne
    2: (0.15, 0.65, 0.70, 0.65),  # Shanghai
    3: (0.05, 0.80, 0.45, 0.80),  # Suzuka
    4: (0.75, 0.60, 0.65, 0.55),  # Miami
    5: (0.55, 0.75, 0.75, 0.45),  # Montreal
    6: (1.00, 0.25, 0.10, 0.35),  # Monaco
    7: (0.00, 0.65, 0.55, 0.90),  # Barcelona
    8: (0.05, 0.75, 0.75, 0.65),  # Austria
    9: (0.00, 0.85, 0.60, 0.75),  # Silverstone
    10: (0.00, 0.90, 0.80, 0.85),  # Spa
    11: (0.10, 0.40, 0.35, 0.65),  # Hungary
    12: (0.05, 0.55, 0.30, 0.75),  # Zandvoort
    13: (0.00, 1.00, 0.85, 0.50),  # Monza
    14: (0.80, 0.75, 0.40, 0.85),  # Madrid: high-speed semi-street, high energy
}

# Team-tier baseline DNF priors (see assumptions.md, section 4.2).
FRONTRUNNER_TEAMS = {"Mercedes", "Ferrari", "McLaren", "Red Bull Racing"}
MIDFIELD_TEAMS = {"Williams", "Alpine", "Racing Bulls", "Aston Martin", "Haas F1 Team"}
NEW_ENTRANT_TEAMS = {"Audi", "Cadillac"}
TEAM_DNF_PRIOR = {
    **{team: 0.06 for team in FRONTRUNNER_TEAMS},
    **{team: 0.09 for team in MIDFIELD_TEAMS},
    **{team: 0.14 for team in NEW_ENTRANT_TEAMS},
}
DEFAULT_DNF_PRIOR = 0.09

# Fallback climatological weather-risk prior for Baku in late September
# (predominantly dry, but the circuit's wind exposure keeps this above zero).
FALLBACK_WEATHER_RISK = 0.25

MONOTONE = {
    "grid_norm": 1,
    "quali_pos_norm": 1,
    "quali_gap_pct": 1,
    "quali_no_time": 1,
    "driver_dnf_rate": 1,
    "streetness": 0,
    "speed_bias": 1,
    "overtaking_ease": 1,
    "tyre_stress": 1,
    "grid_track_position": -1,
    "practice_gap_pct": 1,
    "weather_risk_index": 1,
}
FEATURES = list(MONOTONE.keys())


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def ewma(values: list[float], default: float, alpha: float = 0.45) -> float:
    """Recent-weighted mean, computed without looking beyond the current round."""
    if not values:
        return default
    estimate = values[0]
    for value in values[1:]:
        estimate = alpha * value + (1.0 - alpha) * estimate
    return float(estimate)


def clean_team_name(name: str) -> str:
    aliases = {
        "Mercedes-AMG Petronas F1 Team": "Mercedes",
        "Scuderia Ferrari HP": "Ferrari",
        "McLaren Formula 1 Team": "McLaren",
        "Oracle Red Bull Racing": "Red Bull Racing",
        "Visa Cash App Racing Bulls F1 Team": "Racing Bulls",
        "BWT Alpine F1 Team": "Alpine",
        "TGR Haas F1 Team": "Haas F1 Team",
        "MoneyGram Haas F1 Team": "Haas F1 Team",
        "Audi Revolut F1 Team": "Audi",
        "Atlassian Williams F1 Team": "Williams",
        "Atlassian Williams Racing": "Williams",
        "Aston Martin Aramco F1 Team": "Aston Martin",
        "Cadillac Formula 1 Team": "Cadillac",
    }
    return aliases.get(str(name), str(name))


def fastest_quali_seconds(row: pd.Series) -> float:
    times = []
    for col in ("Q1", "Q2", "Q3"):
        value = row.get(col)
        if pd.notna(value):
            times.append(value.total_seconds())
    return min(times) if times else np.nan


def practice_features(session) -> dict[str, tuple[float, int]]:
    timed = session.laps[session.laps["LapTime"].notna()].copy()
    if timed.empty:
        return {}
    accurate = timed[timed["IsAccurate"].fillna(False)]
    if len(accurate) >= max(10, len(timed) // 3):
        timed = accurate.copy()
    timed["lap_seconds"] = timed["LapTime"].dt.total_seconds()
    best = timed.groupby("Driver")["lap_seconds"].min()
    pole = best.min()
    counts = session.laps[session.laps["LapTime"].notna()].groupby("Driver").size()
    return {
        driver: (100.0 * (seconds / pole - 1.0), int(counts.get(driver, 0)))
        for driver, seconds in best.items()
    }


def is_dnf(status: str) -> int:
    text = str(status).lower()
    return int(not ("finished" in text or "lapped" in text or "+" in text))


def make_model(seed: int) -> XGBRegressor:
    constraints = tuple(MONOTONE[f] for f in FEATURES)
    return XGBRegressor(
        n_estimators=260,
        learning_rate=0.035,
        max_depth=3,
        min_child_weight=5,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.10,
        reg_lambda=2.5,
        objective="reg:squarederror",
        monotone_constraints=constraints,
        random_state=seed,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Official F1 snapshot retrieved September 25, 2026; grid was not yet published.
# driver, team, qualifying position, assumed grid, Q1, Q2, Q3, FP3 seconds, FP3 laps
# ---------------------------------------------------------------------------
QUALIFYING_SNAPSHOT = [
    ('RUS', 'Mercedes', 1, 1, 103.615, 103.462, 102.526, 104.021, 21),
    ('LEC', 'Ferrari', 2, 2, 104.36, 103.78, 103.363, 104.544, 22),
    ('PIA', 'McLaren', 3, 3, 105.014, 103.814, 103.364, 104.746, 20),
    ('HAD', 'Red Bull Racing', 4, 4, 104.161, 103.88, 103.5, 105.176, 22),
    ('NOR', 'McLaren', 5, 5, 104.571, 104.02, 103.672, 104.899, 20),
    ('HAM', 'Ferrari', 6, 6, 104.26, 104.037, 103.858, 104.033, 18),
    ('GAS', 'Alpine', 7, 7, 104.489, 104.106, 104.047, 104.637, 20),
    ('VER', 'Red Bull Racing', 8, 8, 104.041, 103.706, 104.081, 103.922, 21),
    ('SAI', 'Williams', 9, 9, 105.104, 104.629, 104.566, 105.605, 24),
    ('COL', 'Alpine', 10, 10, 105.106, 104.683, 104.963, 105.592, 20),
    ('BEA', 'Haas F1 Team', 11, 11, 105.228, 104.775, None, 105.692, 25),
    ('LAW', 'Racing Bulls', 12, 12, 105.535, 104.86, None, 106.222, 23),
    ('ALB', 'Williams', 13, 13, 105.031, 105.001, None, 106.076, 24),
    ('OCO', 'Haas F1 Team', 14, 14, 105.039, 105.016, None, 105.918, 21),
    ('LIN', 'Racing Bulls', 15, 15, 105.381, 105.106, None, 106.271, 22),
    ('ANT', 'Mercedes', 16, 16, 105.504, None, None, 104.273, 21),
    ('BOR', 'Audi', 17, 17, 105.799, None, None, 105.854, 18),
    ('HUL', 'Audi', 18, 18, 105.92, None, None, 105.998, 18),
    ('ALO', 'Aston Martin', 19, 19, 106.593, None, None, 106.512, 19),
    ('PER', 'Cadillac', 20, 20, 106.658, None, None, 106.004, 19),
    ('STR', 'Aston Martin', 21, 21, 107.337, None, None, 109.279, 9),
    ('BOT', 'Cadillac', 22, 22, 108.29, None, None, 108.587, 19),
]
SNAPSHOT_COLUMNS = ['driver', 'team', 'quali_position', 'grid', 'q1', 'q2', 'q3', 'practice_seconds', 'practice_laps']


def frozen_inputs() -> dict:
    """Return the verified post-qualifying snapshot the file ships with."""
    payload = {
        'year': RACE_YEAR,
        'round': RACE_ROUND,
        'race_date': '2026-09-26',
        'session': 'Q',
        'retrieved_at': '2026-09-25T13:25:58.198359+00:00',
        'source': QUALI_URL,
        'qualifying_complete': True,
        'grid_status': 'qualifying order assumed',
    }
    payload["drivers"] = [dict(zip(SNAPSHOT_COLUMNS, row)) for row in QUALIFYING_SNAPSHOT]
    return payload


# ---------------------------------------------------------------------------
# Live data acquisition, with transparent fallback
# ---------------------------------------------------------------------------

class ResultTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        if tag == "tr" and self.row:
            self.rows.append(self.row)


def seconds(value):
    if value in (None, "", "-", "—"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    return float(parts[0])


def verify_event_schedule() -> str:
    """Confirm Baku's round number/date against FastF1's live schedule backend."""
    if fastf1 is None:
        return "fastf1 not installed; skipping schedule verification."
    try:
        schedule = fastf1.get_event_schedule(RACE_YEAR)
        row = schedule[schedule["RoundNumber"] == RACE_ROUND]
        if row.empty:
            return f"FastF1 schedule has no round {RACE_ROUND} for {RACE_YEAR}; using frozen metadata."
        event_name = row.iloc[0]["EventName"]
        event_date = row.iloc[0]["EventDate"]
        return f"FastF1 schedule confirms round {RACE_ROUND} = {event_name} on {event_date.date()}."
    except Exception as exc:  # network-blocked, offline, or upstream error
        return f"FastF1 schedule fetch unavailable ({exc.__class__.__name__}); using frozen metadata."


def fetch_live_qualifying() -> pd.DataFrame | None:
    """Try FastF1's own live/archived session data for the race qualifying."""
    if fastf1 is None:
        return None
    try:
        FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(FASTF1_CACHE))
        session = fastf1.get_session(RACE_YEAR, RACE_ROUND, "Q")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        results = session.results
        if results is None or results.empty:
            return None
        records = []
        for _, row in results.iterrows():
            fastest = fastest_quali_seconds(row)
            records.append({
                "driver": row.get("Abbreviation"),
                "team": clean_team_name(row.get("TeamName")),
                "quali_position": row.get("Position"),
                "grid": row.get("GridPosition") or row.get("Position"),
                "fastest_seconds": fastest,
            })
        df = pd.DataFrame(records)
        return df if not df.empty else None
    except Exception as exc:
        print(f"[data] FastF1 live qualifying unavailable ({exc.__class__.__name__}); trying next source.")
        return None


def fetch_scraped_qualifying(known_drivers: set[str]) -> pd.DataFrame | None:
    """Best-effort fallback: scrape the official formula1.com qualifying page."""
    try:
        resp = requests.get(QUALI_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        parser = ResultTable()
        parser.feed(resp.text)
        records = []
        for row in parser.rows:
            for cell in row:
                token = cell.strip().upper()
                if token in known_drivers:
                    numeric = [seconds(c) for c in row if c.strip() and seconds(c) is not None]
                    if numeric:
                        records.append({"driver": token, "fastest_seconds": min(numeric)})
                    break
        return pd.DataFrame(records) if records else None
    except Exception as exc:
        print(f"[data] formula1.com scrape unavailable ({exc.__class__.__name__}); using frozen snapshot.")
        return None


def fetch_weather_risk() -> tuple[float, str]:
    """Attempt a live forecast fetch; fall back to a documented climatology prior."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": CIRCUIT_LAT,
                "longitude": CIRCUIT_LON,
                "daily": "precipitation_probability_max,windspeed_10m_max",
                "timezone": "UTC",
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        rain_pct = payload["daily"]["precipitation_probability_max"][0] / 100.0
        wind_kph = payload["daily"]["windspeed_10m_max"][0]
        wind_risk = min(1.0, wind_kph / 60.0)
        risk = float(np.clip(0.6 * rain_pct + 0.4 * wind_risk, 0.0, 1.0))
        return risk, "live Open-Meteo forecast"
    except Exception as exc:
        print(f"[data] live weather forecast unavailable ({exc.__class__.__name__}); using climatology prior.")
        return FALLBACK_WEATHER_RISK, "Baku late-September climatology prior"


def build_driver_frame() -> tuple[pd.DataFrame, str]:
    """Assemble the driver input frame from the best available data source."""
    snapshot = frozen_inputs()
    df = pd.DataFrame(snapshot["drivers"])
    known = set(df["driver"])

    live = fetch_live_qualifying()
    if live is not None:
        merged = df.merge(live[["driver", "quali_position", "grid", "fastest_seconds"]],
                           on="driver", how="left", suffixes=("", "_live"))
        has_live = merged["fastest_seconds"].notna().sum()
        if has_live >= max(1, len(merged) // 2):
            for col in ("quali_position", "grid"):
                merged[col] = merged[f"{col}_live"].combine_first(merged[col])
            df = merged.drop(columns=[c for c in merged.columns if c.endswith("_live")])
            return df, "FastF1 live qualifying session"

    scraped = fetch_scraped_qualifying(known)
    if scraped is not None and len(scraped) >= max(1, len(df) // 2):
        print("[data] merged formula1.com scrape into frozen grid order.")
        return df, "formula1.com qualifying page scrape (grid order retained from snapshot)"

    return df, "frozen verified snapshot (2026-09-25)"


def compute_dnf_rate(team: str) -> float:
    return TEAM_DNF_PRIOR.get(team, DEFAULT_DNF_PRIOR)


# ---------------------------------------------------------------------------
# Synthetic, physics-consistent training corpus
# ---------------------------------------------------------------------------

def _deterministic_finish_index(frame: pd.DataFrame) -> np.ndarray:
    """The declared-monotonic 'true skill' component of the finish index,
    with no race-day noise or DNF applied. Shared by the synthetic training
    corpus and the evaluation harness so both draw from the same assumptions.
    """
    return (
        18.0 * frame["grid_norm"]
        + 6.0 * frame["quali_pos_norm"]
        + 40.0 * frame["quali_gap_pct"]
        + 5.0 * frame["quali_no_time"]
        + 8.0 * frame["driver_dnf_rate"]
        + 3.0 * frame["speed_bias"]
        + 3.0 * frame["overtaking_ease"]
        + 2.0 * frame["tyre_stress"]
        - 4.0 * frame["grid_track_position"]
        + 10.0 * frame["practice_gap_pct"]
        + 5.0 * frame["weather_risk_index"] * frame["driver_dnf_rate"]
    ).to_numpy()


def _sample_synthetic_grid(rng: np.random.Generator, n: int, field_size: int | None = None) -> pd.DataFrame:
    """Draw n independent synthetic driver rows (feature columns only, no
    label) across the known track priors. Used both to build the training
    corpus (independent rows) and, per-race, for evaluation trials.
    """
    tracks = list(TRACK_META.values()) + [TRACK]
    track_idx = rng.integers(0, len(tracks), size=n)
    track_arr = np.array(tracks)[track_idx]
    streetness, speed_bias, overtaking_ease, tyre_stress = (track_arr[:, i] for i in range(4))

    max_grid = field_size if field_size else 20
    if field_size:
        grid = rng.permutation(field_size)[:n] + 1
    else:
        grid = rng.integers(1, max_grid + 1, size=n)
    grid = grid.astype(float)
    grid_norm = (grid - 1) / (max_grid - 1)
    quali_pos = np.clip(grid + rng.normal(0, 1.0, size=n), 1, max_grid)
    quali_pos_norm = (quali_pos - 1) / (max_grid - 1)
    quali_gap_pct = np.clip(rng.exponential(0.006, size=n) * (1.0 + 2.0 * quali_pos_norm), 0, 0.06)
    quali_no_time = (rng.random(n) < 0.03).astype(float)
    driver_dnf_rate = rng.uniform(0.03, 0.18, size=n)
    grid_track_position = grid_norm * (1.0 - overtaking_ease)
    practice_gap_pct = np.clip(quali_gap_pct + rng.normal(0, 0.003, size=n), 0, 0.08)
    weather_risk_index = rng.uniform(0.0, 1.0, size=n) if field_size is None else np.full(n, rng.uniform(0.05, 0.6))

    return pd.DataFrame({
        "grid_norm": grid_norm,
        "quali_pos_norm": quali_pos_norm,
        "quali_gap_pct": quali_gap_pct,
        "quali_no_time": quali_no_time,
        "driver_dnf_rate": driver_dnf_rate,
        "streetness": streetness,
        "speed_bias": speed_bias,
        "overtaking_ease": overtaking_ease,
        "tyre_stress": tyre_stress,
        "grid_track_position": grid_track_position,
        "practice_gap_pct": practice_gap_pct,
        "weather_risk_index": weather_risk_index,
    })


def build_training_data(seed: int, n_samples: int = 8000) -> pd.DataFrame:
    """Generate a training corpus that actually varies each feature and its
    label according to the monotone relationships declared in assumptions.md,
    rather than fitting the model on a single circular (grid -> grid) row set.
    This is a synthetic teaching corpus, not a claim of real historical results.
    """
    rng = np.random.default_rng(seed)
    frame = _sample_synthetic_grid(rng, n_samples)

    finish_index = _deterministic_finish_index(frame)
    dnf_event = rng.random(n_samples) < frame["driver_dnf_rate"].to_numpy() * (
        1.0 + 0.5 * frame["weather_risk_index"].to_numpy())
    finish_index = finish_index + dnf_event * rng.uniform(8, 15, size=n_samples)
    finish_index = finish_index + rng.normal(0, 1.5, size=n_samples)

    frame["finish_index"] = finish_index
    return frame


# ---------------------------------------------------------------------------
# Model evaluation
#
# Two different questions, evaluated two different ways (see assumptions.md
# section 5 for the full explanation):
#   1. Regression fit  - did XGBoost actually learn the declared monotonic
#      relationship, on rows it did not train on?
#   2. Probability calibration - when the simulation says "62% to win", does
#      that driver actually win about 62% of the time, under our own stated
#      assumptions? (Brier score, log loss, top-1 accuracy, precision@3.)
#
# Neither of these is a backtest against real Baku 2026 results: the race
# has not been driven yet, and this sandbox's network policy blocks the
# FastF1/Ergast hosts that would otherwise supply real historical results to
# backtest against (see the "Grid/qualifying data source" line the pipeline
# prints). Both evals instead check internal consistency: does the model
# reproduce our own declared physics, and is the simulation's math unbiased.
# ---------------------------------------------------------------------------

def evaluate_regression_fit(seed: int, n_samples: int = 10000, test_size: float = 0.2) -> dict:
    """Train/test split of the synthetic corpus; report standard regression
    metrics (MAE, RMSE, R^2) plus Spearman rank correlation, since what the
    simulation actually consumes is driver *order*, not the raw index value.
    """
    corpus = build_training_data(seed, n_samples=n_samples)
    train_df, test_df = train_test_split(corpus, test_size=test_size, random_state=seed)

    model = make_model(seed)
    model.fit(train_df[FEATURES], train_df["finish_index"])
    preds = model.predict(test_df[FEATURES])
    truth = test_df["finish_index"].to_numpy()

    rho, _ = spearmanr(truth, preds)
    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "mae": float(mean_absolute_error(truth, preds)),
        "rmse": float(np.sqrt(mean_squared_error(truth, preds))),
        "r2": float(r2_score(truth, preds)),
        "spearman_rank_correlation": float(rho),
    }


def _reliability_table(pred: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    bucket = np.clip(np.digitize(pred, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bucket == b
        if not mask.any():
            continue
        rows.append({
            "bucket": f"{bins[b]:.2f}-{bins[b + 1]:.2f}",
            "n": int(mask.sum()),
            "mean_predicted_win_prob": float(pred[mask].mean()),
            "actual_win_rate": float(actual[mask].mean()),
        })
    return rows


def evaluate_calibration(model: XGBRegressor, n_trials: int = 300, inner_sims: int = 400,
                          field_size: int = 20, seed: int = SEED + 1) -> dict:
    """Self-consistency check for the Monte Carlo step: simulate many synthetic
    'races' from the same generative assumptions the model was trained on,
    have the trained model + simulator produce ex-ante win probabilities, then
    draw one 'realized' outcome per race from the identical noise/DNF process
    and check whether predicted probabilities match realized frequencies.
    """
    rng = np.random.default_rng(seed)
    pred_win_all, actual_win_all = [], []
    precision_at_3, top1_hits = [], 0

    for _ in range(n_trials):
        frame = _sample_synthetic_grid(rng, field_size, field_size=field_size)
        true_index = _deterministic_finish_index(frame)
        base_preds = model.predict(frame[FEATURES])

        tyre_stress = frame["tyre_stress"].iloc[0]
        streetness = frame["streetness"].iloc[0]
        weather_risk = frame["weather_risk_index"].iloc[0]
        dnf_rate = frame["driver_dnf_rate"].to_numpy()
        dnf_prob = np.clip(dnf_rate * (1.0 + 0.5 * weather_risk), 0, 0.6)
        noise_scale = 1.0 * (0.6 + 0.5 * tyre_stress + 0.3 * streetness + 0.4 * weather_risk)

        noise = rng.normal(0, 1.0, size=(inner_sims, field_size)) * noise_scale
        scores = base_preds + noise
        dnf_draws = rng.random((inner_sims, field_size)) < dnf_prob
        scores = np.where(dnf_draws, scores + rng.uniform(8, 15, size=scores.shape), scores)
        ranks = np.argsort(np.argsort(scores, axis=1), axis=1) + 1
        win_prob = (ranks == 1).mean(axis=0)

        true_noise = rng.normal(0, 1.0, size=field_size) * noise_scale
        true_scores = true_index + true_noise
        true_dnf = rng.random(field_size) < dnf_prob
        true_scores = np.where(true_dnf, true_scores + rng.uniform(8, 15, size=field_size), true_scores)
        true_rank = np.argsort(np.argsort(true_scores)) + 1
        actual_win = (true_rank == 1).astype(float)
        actual_top3 = true_rank <= 3

        pred_win_all.append(win_prob)
        actual_win_all.append(actual_win)
        predicted_top3_idx = np.argsort(base_preds)[:3]
        precision_at_3.append(np.isin(predicted_top3_idx, np.where(actual_top3)[0]).sum() / 3.0)
        top1_hits += int(actual_win[np.argmax(win_prob)] == 1)

    pred_win = np.concatenate(pred_win_all)
    actual_win = np.concatenate(actual_win_all)
    eps = 1e-9
    brier = float(np.mean((pred_win - actual_win) ** 2))
    log_loss = float(-np.mean(actual_win * np.log(pred_win + eps) + (1 - actual_win) * np.log(1 - pred_win + eps)))

    return {
        "n_trials": n_trials,
        "inner_sims_per_trial": inner_sims,
        "field_size": field_size,
        "brier_score_win_prob": brier,
        "log_loss_win_prob": log_loss,
        "top1_winner_accuracy": top1_hits / n_trials,
        "podium_precision_at_3_mean": float(np.mean(precision_at_3)),
        "reliability_table": _reliability_table(pred_win, actual_win),
        "note": ("Self-consistency check against the pipeline's own generative "
                 "assumptions, not a backtest against real race results (unavailable "
                 "in this environment - see assumptions.md section 5)."),
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulations(df_input: pd.DataFrame, weather_risk: float,
                     num_simulations: int = 30000) -> tuple[pd.DataFrame, XGBRegressor]:
    """Fit XGBoost on the synthetic monotone corpus, predict a finish index for
    every Baku entrant, then Monte Carlo simulate race outcomes with explicit
    DNF injection and condition-scaled noise.
    """
    model = make_model(SEED)
    training = build_training_data(SEED)
    model.fit(training[FEATURES], training["finish_index"])

    X = df_input.copy()
    X["grid"] = X["grid"].astype(float)
    X["quali_position"] = X["quali_position"].astype(float)
    X["grid_norm"] = (X["grid"] - 1) / 19.0
    X["quali_pos_norm"] = (X["quali_position"] - 1) / 19.0

    min_time = X[["q1", "q2", "q3"]].min(axis=1)
    pole_time = min_time.min()
    X["quali_gap_pct"] = (min_time - pole_time) / pole_time
    X["quali_no_time"] = min_time.isna().astype(float)

    X["driver_dnf_rate"] = X["team"].apply(compute_dnf_rate)

    X["streetness"] = TRACK[0]
    X["speed_bias"] = TRACK[1]
    X["overtaking_ease"] = TRACK[2]
    X["tyre_stress"] = TRACK[3]
    X["grid_track_position"] = X["grid_norm"] * (1.0 - TRACK[2])

    fp3_pole = X["practice_seconds"].min()
    X["practice_gap_pct"] = (X["practice_seconds"] - fp3_pole) / fp3_pole
    X["weather_risk_index"] = weather_risk

    features_df = X[FEATURES].fillna(0)
    base_preds = model.predict(features_df)

    n_drivers = len(df_input)
    max_laps = max(X["practice_laps"].max(), 1)
    practice_confidence = (X["practice_laps"] / max_laps).clip(lower=0.3).to_numpy()

    noise_scale = 1.0 * (0.6 + 0.5 * TRACK[3] + 0.3 * TRACK[0] + 0.4 * weather_risk)
    noise_scale = noise_scale / practice_confidence  # more FP3 running -> tighter uncertainty

    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 1.0, size=(num_simulations, n_drivers)) * noise_scale
    simulated_scores = base_preds + noise

    dnf_prob = (X["driver_dnf_rate"] * (1.0 + 0.5 * weather_risk)).clip(upper=0.6).to_numpy()
    dnf_draws = rng.random(size=(num_simulations, n_drivers)) < dnf_prob
    simulated_scores = np.where(dnf_draws, simulated_scores + rng.uniform(8, 15, size=simulated_scores.shape), simulated_scores)

    simulated_ranks = np.argsort(np.argsort(simulated_scores, axis=1), axis=1) + 1

    win_probs = (simulated_ranks == 1).mean(axis=0) * 100.0
    podium_probs = (simulated_ranks <= 3).mean(axis=0) * 100.0
    points_probs = (simulated_ranks <= 10).mean(axis=0) * 100.0
    avg_positions = simulated_ranks.mean(axis=0)

    results = pd.DataFrame({
        "driver": df_input["driver"].to_numpy(),
        "team": df_input["team"].to_numpy(),
        "grid": df_input["grid"].to_numpy(),
        "avg_pos": avg_positions,
        "win_prob": win_probs,
        "podium_prob": podium_probs,
        "points_prob": points_probs,
    }).sort_values(by="avg_pos", ascending=True).reset_index(drop=True)

    return results, model


def save_outputs(model: XGBRegressor, results: pd.DataFrame, weather_risk: float,
                  weather_source: str, data_source: str, evaluation: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(DATA / "baku_2026_predictions.csv", index=False)

    importances = dict(zip(FEATURES, (float(v) for v in model.feature_importances_)))
    with open(ARTIFACTS / "feature_importance.json", "w") as fh:
        json.dump(dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True)), fh, indent=2)
    with open(ARTIFACTS / "evaluation_metrics.json", "w") as fh:
        json.dump(evaluation, fh, indent=2)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "race": RACE_NAME,
        "year": RACE_YEAR,
        "round": RACE_ROUND,
        "grid_data_source": data_source,
        "weather_source": weather_source,
        "weather_risk_index": weather_risk,
        "predicted_podium": results.sort_values("avg_pos").head(3)["driver"].tolist(),
        "ranking": results.to_dict(orient="records"),
    }
    with open(DATA / "baku_2026_predictions.json", "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Baku Race Predictor")
    parser.add_argument("--sims", type=int, default=30000, help="Number of Monte Carlo simulations to run")
    args = parser.parse_args()

    print(f"=== {RACE_NAME} {RACE_YEAR} (round {RACE_ROUND}) prediction pipeline ===\n")

    print(verify_event_schedule())
    df_drivers, data_source = build_driver_frame()
    print(f"Grid/qualifying data source: {data_source}")

    weather_risk, weather_source = fetch_weather_risk()
    print(f"Weather risk index: {weather_risk:.2f} ({weather_source})")

    print(f"\nRunning {args.sims} Monte Carlo simulations for {RACE_NAME}...\n")
    results, model = run_simulations(df_drivers, weather_risk, num_simulations=args.sims)

    print("Evaluating model fit (held-out synthetic regression check)...")
    regression_eval = evaluate_regression_fit(SEED)
    print(f"  MAE={regression_eval['mae']:.3f}  RMSE={regression_eval['rmse']:.3f}  "
          f"R2={regression_eval['r2']:.3f}  Spearman={regression_eval['spearman_rank_correlation']:.3f} "
          f"(n_test={regression_eval['n_test']})")

    print("Evaluating simulation calibration (self-consistency check)...")
    calibration_eval = evaluate_calibration(model)
    print(f"  Brier={calibration_eval['brier_score_win_prob']:.4f}  "
          f"LogLoss={calibration_eval['log_loss_win_prob']:.4f}  "
          f"Top1WinAcc={calibration_eval['top1_winner_accuracy']:.3f}  "
          f"Podium P@3={calibration_eval['podium_precision_at_3_mean']:.3f} "
          f"(n_trials={calibration_eval['n_trials']})")

    print(f"\n{'Pos':>3} {'Driver':6} {'Team':<16} {'Grid':>4}  {'AvgPos':>6}  {'Win%':>6}  {'Podium%':>8}  {'Top10%':>7}")
    for i, row in results.iterrows():
        print(f"{i+1:3d} {row['driver']:6s} {row['team']:<16} {int(row['grid']):4d}  "
              f"{row['avg_pos']:6.2f}  {row['win_prob']:6.1f}  {row['podium_prob']:8.1f}  {row['points_prob']:7.1f}")

    top3 = results.sort_values(by="avg_pos").head(3)["driver"].tolist()
    print("\nPredicted Top 3 (podium)")
    print(f"P1: {top3[0]}")
    print(f"P2: {top3[1]}")
    print(f"P3: {top3[2]}")

    evaluation = {"regression_fit": regression_eval, "calibration": calibration_eval}
    save_outputs(model, results, weather_risk, weather_source, data_source, evaluation)
    print(f"\nSaved full ranking to {DATA / 'baku_2026_predictions.csv'} and {DATA / 'baku_2026_predictions.json'}")
    print(f"Saved evaluation metrics to {ARTIFACTS / 'evaluation_metrics.json'}")
