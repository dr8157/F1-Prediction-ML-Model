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
