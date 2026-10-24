# F1 Virtual Race Engineer

A self-updating ML pipeline that predicts F1 lap times, detects drift as new
race data arrives, retrains candidate models, and promotes a candidate only
if it beats the current model on a held-out validation race.

## Pipeline
1. `main.py` downloads race data with FastF1 (Bahrain, Monza, Silverstone,
   2021-2024), cleans it, and saves `data/processed_laps_multi.csv`.
2. `train.py` trains a baseline RandomForest with a time-based split (train on
   2021, test on 2022-2024) and compares it to a naive baseline.
3. `pipeline.py` runs the self-updating loop:
   - Races arrive in chronological order.
   - Drift is flagged when the deployed model's MAE on a new race exceeds 2x a
     1.0 s baseline.
   - On drift, a candidate is trained on all history so far.
   - Candidate and current models are compared on a fixed held-out validation
     race (2024 Silverstone), which is never trained on, to avoid leakage.
   - The candidate is promoted only if it is better; otherwise the current
     model is kept (rollback).
   - Every model version is saved with metadata in `models/registry.json`.

## Setup
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    mkdir data\cache
    python main.py
    python train.py
    python pipeline.py

Note: `data/cache/` must exist before running `main.py`. The `.joblib` model
files are not in the repo; running `pipeline.py` regenerates them.

## Limitations
- Only 3 circuits are used.
- Races within a season are ordered alphabetically, not by real calendar date.
- The drift threshold is hand-picked.
