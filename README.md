# F1-Prediction-ML-Model
ML model to predict the winners of different F1 racing

## Baku Grand Prix predictor (`prediction9.py`)

XGBoost (monotonic constraints) + Monte Carlo simulation pipeline predicting
the full-field finishing order for the 2026 Azerbaijan Grand Prix.

```bash
pip install -r requirements.txt
python3 prediction9.py --sims 30000
```

The script tries live data first (FastF1 schedule/session, then a
formula1.com results-page scrape) and falls back transparently to a verified
frozen qualifying snapshot when those sources aren't reachable (e.g. a
sandboxed network policy), printing which source was actually used. See
`assumptions.md` for the full feature/model/assumption documentation.

Outputs:
* Console: full-field ranking (avg finish position, win %, podium %, top-10 %) and predicted top 3.
* `data/baku_2026_predictions.csv` / `.json`: the same ranking, saved.
* `data/baku_model/feature_importance.json`: trained model's feature importances.

### How it works

1. **Data acquisition** (`verify_event_schedule`, `fetch_live_qualifying`, `fetch_scraped_qualifying`, `fetch_weather_risk`): tries FastF1's live schedule/session data and a formula1.com scrape first, and falls back to the verified frozen qualifying snapshot (`QUALIFYING_SNAPSHOT`, retrieved 2026‑09‑25) whenever those hosts aren't reachable — each run prints which source was actually used.
2. **Feature engineering** (`run_simulations`): builds the 12 features defined in `assumptions.md` — grid/qualifying normalization, qualifying gap %, team-tiered `driver_dnf_rate`, the four Baku track priors (`streetness`, `speed_bias`, `overtaking_ease`, `tyre_stress`), the `grid_track_position` interaction term, plus two added features: `practice_gap_pct` (FP3 pace) and `weather_risk_index` (live forecast or climatology fallback).
3. **Model** (`make_model`, `build_training_data`): an `XGBRegressor` with monotone constraints matching the racing-physics direction of each feature, trained on an 8,000-row synthetic corpus generated from those same monotonic relationships (replacing the original circular grid→grid fit).
4. **Monte Carlo simulation** (`run_simulations`): 30,000 simulated races per run, each drawing a per-driver DNF event (probability from `driver_dnf_rate`, amplified by weather risk) and condition-scaled noise, then ranking the field to get win / podium / points-finish probabilities for every driver.

### Latest run — Azerbaijan GP 2026, 30,000 simulations

Data source: frozen verified snapshot (FastF1 live session and formula1.com scrape unavailable). Weather risk index: 0.25 (Baku late-September climatology prior).

| Pos | Driver | Team | Grid | Avg. finish | Win % | Podium % | Top-10 % |
|----:|:------:|------|:----:|-----------:|------:|---------:|---------:|
| 1 | RUS | Mercedes | 1 | 1.72 | **74.2** | 92.8 | 99.3 |
| 2 | LEC | Ferrari | 2 | 2.88 | 14.3 | 82.6 | 97.9 |
| 3 | PIA | McLaren | 3 | 3.19 | 10.0 | 75.1 | 97.5 |
| 4 | HAD | Red Bull Racing | 4 | 4.61 | 0.9 | 25.2 | 95.8 |
| 5 | NOR | McLaren | 5 | 4.88 | 0.6 | 19.2 | 95.5 |
| 6 | HAM | Ferrari | 6 | 6.06 | 0.1 | 4.7 | 94.2 |
| 7 | GAS | Alpine | 7 | 7.50 | 0.0 | 0.3 | 90.6 |
| 8 | VER | Red Bull Racing | 8 | 7.75 | 0.0 | 0.1 | 93.5 |
| 9 | SAI | Williams | 9 | 9.19 | 0.0 | 0.0 | 89.0 |
| 10 | COL | Alpine | 10 | 10.41 | 0.0 | 0.0 | 73.2 |
| 11 | BEA | Haas F1 Team | 11 | 11.13 | 0.0 | 0.0 | 49.7 |
| 12 | LAW | Racing Bulls | 12 | 12.41 | 0.0 | 0.0 | 15.4 |
| 13 | ALB | Williams | 13 | 13.13 | 0.0 | 0.0 | 5.9 |
| 14 | OCO | Haas F1 Team | 14 | 14.00 | 0.0 | 0.0 | 2.1 |
| 15 | LIN | Racing Bulls | 15 | 14.88 | 0.0 | 0.0 | 0.4 |
| 16 | ANT | Mercedes | 16 | 15.40 | 0.0 | 0.0 | 0.1 |
| 17 | BOR | Audi | 17 | 17.40 | 0.0 | 0.0 | 0.0 |
| 18 | ALO | Aston Martin | 19 | 18.25 | 0.0 | 0.0 | 0.0 |
| 19 | HUL | Audi | 18 | 18.33 | 0.0 | 0.0 | 0.0 |
| 20 | STR | Aston Martin | 21 | 19.54 | 0.0 | 0.0 | 0.0 |
| 21 | PER | Cadillac | 20 | 19.91 | 0.0 | 0.0 | 0.0 |
| 22 | BOT | Cadillac | 22 | 20.44 | 0.0 | 0.0 | 0.0 |

**Predicted podium: 🥇 RUS — 🥈 LEC — 🥉 PIA**

Full machine-readable output: [`data/baku_2026_predictions.csv`](data/baku_2026_predictions.csv) / [`data/baku_2026_predictions.json`](data/baku_2026_predictions.json).
