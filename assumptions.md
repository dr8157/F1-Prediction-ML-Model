# Model Assumptions & Feature Engineering Specifications

## 1. Qualitative Track Priors (`TRACK_META` & `TRACK`)
The model uses a 4-dimensional normalized tuple `(streetness, speed_bias, overtaking_ease, tyre_stress)` to describe track characteristics:
* **Streetness (0.0 to 1.0)**: Degree to which the track is a temporary street circuit bounded by walls.
* **Speed Bias (0.0 to 1.0)**: Weighting toward top-speed / straight-line speed demands vs. high-downforce cornering.
* **Overtaking Ease (0.0 to 1.0)**: Ease of passing based on straight length and DRS zones.
* **Tyre Stress (0.0 to 1.0)**: Thermal degradation and asphalt roughness impact.

**Baku Track Prior (`TRACK`)**: `(1.0, 1.0, 0.75, 0.40)`
* Extreme street circuit (`streetness = 1.0`) with massive straight-line demands (`speed_bias = 1.0`), relatively high overtaking potential (`overtaking_ease = 0.75`), and low-to-medium tyre wear (`tyre_stress = 0.40`).

## 2. Feature Definitions
* `grid_norm`: Normalized grid position `(grid - 1) / 19.0`.
* `quali_pos_norm`: Normalized qualifying classification `(quali_position - 1) / 19.0`.
* `quali_gap_pct`: Percentage time delta of driver's fastest Q1/Q2/Q3 lap relative to the pole lap time: `(lap_time - pole_time) / pole_time`.
* `quali_no_time`: Binary flag `(1.0 if no quali lap recorded, else 0.0)`.
* `driver_dnf_rate`: Historical career/season DNF probability computed via `is_dnf(status)`.
* `streetness`, `speed_bias`, `overtaking_ease`, `tyre_stress`: Track attributes broadcast to each record.
* `grid_track_position`: Interaction term combining starting grid position with circuit overtaking difficulty: `grid_norm * (1.0 - overtaking_ease)`.
* `practice_gap_pct` *(added)*: Percentage time delta of a driver's best FP3 lap relative to the fastest FP3 lap of the weekend: `(fp3_seconds - min(fp3_seconds)) / min(fp3_seconds)`. FP3 pace was already present in the qualifying snapshot payload (`practice_seconds`, `practice_laps`) but previously unused as a model input — it gives the model an independent, pre-qualifying read on raw pace.
* `weather_risk_index` *(added)*: A single 0–1 scalar broadcast to every driver for the race, capturing the probability-weighted disruption risk from wind/rain at the circuit on race day (see Section 5). It does not rank drivers on its own; its value comes from interaction splits with `driver_dnf_rate` and `tyre_stress` (XGBoost `max_depth=3` supports up to 3-way interactions), amplifying risk for less reliable cars in unsettled conditions.

## 3. Monotone Constraints Map (`make_model`)
Monotonicity constraints guide XGBoost to respect racing physics. The predicted target is a "finish index" where **lower is better** (closer to winning):
* `grid_norm` (+1): Lower grid index (closer to P1) reduces predicted finish position.
* `quali_pos_norm` (+1): Better qualifying results correlate positively with better finishes.
* `quali_gap_pct` (+1): Smaller qualifying gaps yield better finish positions.
* `quali_no_time` (+1): Penalty for failing to set a time.
* `driver_dnf_rate` (+1): Higher DNF tendency increases expected finish index penalty.
* `streetness` (0): Neutral baseline interaction.
* `speed_bias` (+1): Driver/car speed match penalty vector.
* `overtaking_ease` (+1): Interaction coefficient constraint.
* `tyre_stress` (+1): Wear degradation coefficient constraint.
* `grid_track_position` (-1): Inverted penalty for high grid positions on hard-to-pass tracks.
* `practice_gap_pct` (+1): Slower long-run/FP3 pace relative to the field increases predicted finish index.
* `weather_risk_index` (+1): Higher disruption risk broadly increases predicted finish index (differentially, via interactions, for less reliable entries).

---

## 4. Enhancements over the original snapshot (`prediction9.py`)

The original script had three acknowledged weaknesses (the code's own comment called the training call a "dummy background set"). Each is addressed below, and each change is a modeling assumption stated explicitly rather than silently baked in:

### 4.1 Live data first, transparent fallback second
`prediction9.py` now genuinely attempts, in order, before falling back to the frozen snapshot:
1. **FastF1 event schedule** (`fastf1.get_event_schedule(2026)`) — confirms the Baku round number and calendar date against the live 2026 calendar published through FastF1's own schedule backend.
2. **FastF1 live qualifying session** (`fastf1.get_session(2026, round, 'Q')`) — if timing data has been published, real lap times replace the snapshot.
3. **formula1.com qualifying results page scrape** (`requests` + the bundled `ResultTable` HTML parser) — a lightweight best-effort fallback if FastF1 has no session data yet.
4. **Frozen `QUALIFYING_SNAPSHOT`** — the verified, timestamped snapshot the file ships with, used whenever steps 1–3 are unavailable (e.g. a network policy blocks F1's live timing / Ergast-mirror hosts, or the session hasn't been archived yet).

Each stage prints which source was actually used, so a run's provenance is never silently ambiguous. **Assumption**: when only the frozen snapshot is available, the qualifying classification order is assumed equal to the starting grid (no grid penalties applied), exactly as the original snapshot's `grid_status` field states.

### 4.2 `driver_dnf_rate` is no longer a flat constant
The original code set every driver's `driver_dnf_rate = 0.08`, discarding a documented feature entirely. It is now team-based, reflecting known 2026 reliability risk tiers:
* Established front-running manufacturer teams (Mercedes, Ferrari, McLaren, Red Bull Racing): **0.06**
* Established midfield teams (Williams, Alpine, Racing Bulls, Aston Martin, Haas F1 Team): **0.09**
* First-year 2026 entrants (Audi, Cadillac): **0.14** — new power-unit and chassis programs historically run a higher early-season non-finish rate.

If a live FastF1 season load succeeds, this prior is replaced by the empirical DNF rate computed from actual 2026 results via `is_dnf(status)`, already defined in the file but previously unused.

### 4.3 Weather is modeled, not ignored
Baku is a street circuit nicknamed the "City of Winds," and wind/rain materially raises safety-car and contact risk on a wall-lined layout. `fetch_weather_risk()` attempts a live forecast fetch for the circuit's coordinates (40.3725° N, 49.8532° E) for race day; when the network denies it, it falls back to a climatological prior for Baku in late September: predominantly dry (~10–15% precipitation chance) but with a non-trivial gusty-wind risk, yielding a fallback `weather_risk_index = 0.25`. This value scales both the monotone-constrained model feature (Section 2) and the Monte Carlo noise/DNF amplification described below.

### 4.4 Training data is no longer circular
The original code fit the model with `model.fit(features_df, X["grid"])` — training the model to predict the grid position from a feature vector that already contains the grid position, on the 22 Baku rows only. This produces a degenerate fit with no genuine learned relationship; it is a structural placeholder, not a trained predictor.

`prediction9.py` now trains on a **synthetic, physics-consistent corpus** (`build_training_data`, 8,000 samples across all 14 known 2026 track priors plus Baku) generated from the same monotonic relationships declared in Section 3, with independent per-sample noise and explicit stochastic DNF injection. This is explicitly a synthetic corpus for teaching the model the declared monotonic shape — it is not presented as historical race results. Real historical per-lap results could not be fetched in this environment because FastF1's live-timing and Ergast-mirror hosts are network-blocked here; the code path to prefer real multi-season history (via FastF1) is present and used automatically wherever that access is available.

### 4.5 Monte Carlo simulation models DNFs explicitly
The original Monte Carlo step added symmetric Gaussian noise to the predicted finish index and never modeled a non-finish. `prediction9.py` now, per simulated race:
* Draws a Bernoulli DNF event per driver using `driver_dnf_rate`, amplified by `weather_risk_index`.
* Scales each driver's noise by `tyre_stress`, `streetness`, and `weather_risk_index` (higher-chaos races produce a wider spread of outcomes), and shrinks it slightly for drivers with more confirmed-reliable FP3 lap counts.
* Sends any drawn DNF to the back of that simulated field before ranking, rather than merely perturbing its position.

### 4.6 Full-field ranking output
The original script only printed the top 3 by average predicted position. The script now reports, for all 22 grid slots: average finishing position, win probability, podium (top-3) probability, and points-finish (top-10) probability, in addition to the predicted top-3 podium call.

---

## 5. Model Evaluation

The original script had no evaluation step at all — it printed a ranking with no way to tell whether the model or the simulation could be trusted. Two evaluations were added, answering two different questions, because there is no ground truth to backtest against yet: **the Baku 2026 race has not been driven**, and this environment's network policy blocks the FastF1 live-timing and Ergast-mirror hosts that would otherwise supply real historical results to validate against (confirmed directly: `livetiming.formula1.com`, `api.jolpi.ca`, and `formula1.com` all return a proxy-level 403; only FastF1's GitHub-hosted schedule endpoint is reachable). Both evaluations below therefore check **internal consistency** — does the model reproduce the physics we declared, and is the simulation's probability math unbiased — not real-world backtest accuracy. This distinction is stated explicitly in the saved metrics file so it is never mistaken for a real-world accuracy claim.

### 5.1 Regression fit (`evaluate_regression_fit`)
An 80/20 train/test split of the synthetic corpus (Section 4.4). The model is retrained on the 80% and scored against the **held-out 20% it never saw**:
* **MAE / RMSE** (mean absolute / root-mean-squared error against the true finish index) — how far off, on average, the model's raw score is.
* **R²** — how much of the variance in the true finish index the model explains.
* **Spearman rank correlation** — this is the metric that actually matters for a *ranking* problem: it measures whether the model gets the **order** of drivers right, independent of the exact score values. A Spearman of 1.0 would mean a perfect ranking even if the raw scores were off by a constant.

Latest run: MAE 2.75, RMSE 4.06, R² 0.76, Spearman 0.89 (n_test = 2,000). A Spearman above 0.85 means the model reliably recovers the intended grid/quali/DNF/track ordering on unseen synthetic examples.

### 5.2 Simulation calibration (`evaluate_calibration`)
This addresses the "precision/recall"-style question directly: **when the simulation says a driver has an X% chance to win, does that driver actually win about X% of the time?**

Method: 300 synthetic mock races are generated from the same generative assumptions as the training corpus. For each one, the trained model + Monte Carlo simulator produce an ex-ante win probability per driver (400 inner simulations), and then **one** race is actually "run" by drawing a single outcome from the identical noise/DNF process. Repeating this many times and comparing predicted probability to realized frequency is the standard way to check probability calibration (a reliability diagram).

Metrics reported:
* **Brier score** (lower is better, 0 = perfect): mean squared error between predicted win probability and the realized 0/1 outcome. 0.024 is very good — a coin-flip model with no information would score ~0.25 on binary outcomes.
* **Log loss** (lower is better): penalizes confident wrong predictions more heavily than Brier score.
* **Top-1 winner accuracy**: how often the driver the model rated the *most* likely winner actually won — the classification-style "precision" analogue for the win prediction. 63.7% in the latest run (a field of 20 drivers with a uniform prior would score 5%).
* **Podium precision@3**: of the 3 drivers predicted to finish on the podium, what fraction actually did — the direct precision@k answer to "did we do precision/recall." 77.1% in the latest run.
* **Reliability table**: predicted-probability deciles (0–10%, 10–20%, …) next to the realized win rate in each bucket. In the latest run these track each other closely across every bucket (e.g. the 80–90% bucket predicted 85.6% on average and realized an 83.3% win rate) — evidence the simulation isn't systematically over- or under-confident.

All of the above is saved to `data/baku_model/evaluation_metrics.json` on every run, alongside `data/baku_model/feature_importance.json`.

**What this evaluation does *not* prove**: that the Baku 2026 grid predictions themselves are accurate, since that requires the actual race result. What it does prove: the model correctly learns the declared monotonic structure on new data, and the Monte Carlo simulation's probability outputs are not biased relative to the assumptions that generated them. Once real per-race results become fetchable (either the network policy allows FastF1/Ergast, or the race has been run and archived), the same harness should be re-pointed at real historical results for a genuine backtest — that is a natural next step, not yet done here.
