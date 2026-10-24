"""
pipeline.py - Self-updating race engineer pipeline.

Simulates new races arriving in chronological order. For each race it:
  1. measures how well the currently deployed model predicts it (drift check),
  2. if drift is detected, trains a candidate model on all history so far,
  3. compares candidate vs current on a fixed held-out validation race,
  4. promotes the candidate only if it is better, otherwise keeps the
     current model (rollback).
  5. saves every trained model to models/ and records its metadata in
     models/registry.json.

Input:  data/processed_laps_multi.csv (created by main.py)
Output: data/pipeline_log.csv, models/*.joblib, models/registry.json
"""
import os
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Folder and file for saved model versions 
os.makedirs('models', exist_ok=True)
REGISTRY_PATH = 'models/registry.json'

# Load and prepare data
df = pd.read_csv('data/processed_laps_multi.csv')

# Convert categorical text columns into True/False columns so the model can use them
df = pd.get_dummies(df, columns=['Driver', 'Team', 'Compound', 'Circuit'])

# target and feature columns.
target = 'LapTimeSeconds'
drop_cols = ['LapTime', 'PitInTime', 'PitOutTime', target]
feature_cols = [c for c in df.columns if c not in drop_cols]

# get_dummies removed the original Circuit column, so recover the readable circuit name from whichever Circuit_* column is True,
# then build a "Race" label such as "2023 Monza" to identify each race.
circuit_cols = [c for c in df.columns if c.startswith('Circuit_')]
df['CircuitName'] = df[circuit_cols].idxmax(axis=1).str.replace('Circuit_', '')
df['Race'] = df['Season'].astype(str) + ' ' + df['CircuitName']

# Order races chronologically: by season (as a number), then by circuit
# name. Note: within a season this is alphabetical, not the real calendar.
race_order = sorted(df['Race'].unique(), key=lambda r: (int(r.split()[0]), r.split()[1]))
print("Race order:", race_order)

# Fixed validation race
# The last race is held out and NEVER used for training. Every candidate
# and current model is judged on it, so comparisons are fair and free of
# data leakage
VALIDATION_RACE = race_order[-1]
pipeline_races = race_order[:-1]
print(f"Validation race (held out, never trained on): {VALIDATION_RACE}")

val_data = df[df['Race'] == VALIDATION_RACE]

# fucntion to split a table of laps into inputs (X) and the target (y).
def get_features_target(race_subset):
    X = race_subset[feature_cols]
    y = race_subset[target]
    return X, y

X_val, y_val = get_features_target(val_data)

# function to train a fresh RandomForest on the given laps and return it.
def train_model(data_subset):
    X, y = get_features_target(data_subset)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

# Model versioning
# The registry records every model version and which one is deployed.
registry = {'current_version': None, 'versions': []}

# Save a model to disk and record its metadata in the registry.
def save_version(model, version, trained_races, training_rows,
                 validation_mae, status, triggered_by):
    filename = f"models/model_v{version}.joblib"
    joblib.dump(model, filename)          # joblib saves the trained model to a file
    registry['versions'].append({
        'version': version,
        'file': filename,
        'created': datetime.now().isoformat(timespec='seconds'),
        'trained_on_races': list(trained_races),         # which data it saw
        'training_rows': int(training_rows),
        'n_estimators': 200,                             # model configuration
        'validation_mae': float(validation_mae),         # its accuracy
        'status': status,                                # INITIAL / PROMOTED / REJECTED
        'triggered_by': triggered_by,                    # race that caused retraining
    })
    return filename

# Write the registry to models/registry.json.
def save_registry():
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)
# fucntion to reload any previously saved version (manual rollback).
def rollback_to(version):
    for v in registry['versions']:
        if v['version'] == version:
            registry['current_version'] = version
            save_registry()
            return joblib.load(v['file'])
    raise ValueError(f"Version {version} not found")

# Drift settings 
# Drift is flagged when a race's MAE exceeds BASELINE_MAE * DRIFT_MULTIPLIER
# The baseline is a hand-picked value for normal error.
BASELINE_MAE = 1.0
DRIFT_MULTIPLIER = 2.0

# Initial model 
training_data = df[df['Race'] == pipeline_races[0]].copy()
trained_races = [pipeline_races[0]]

current_model = train_model(training_data)
initial_val_mae = mean_absolute_error(y_val, current_model.predict(X_val))

# Save it as version 0 and mark it as the deployed model.
version_counter = 0
save_version(current_model, version_counter, trained_races, len(training_data),
             initial_val_mae, 'INITIAL', 'initial training')
registry['current_version'] = version_counter
save_registry()
print(f"Initial model v0 trained on: {pipeline_races[0]} "
      f"({len(training_data)} rows), validation MAE={initial_val_mae:.3f}")

log = []     # one entry per race, saved as pipeline_log.csv at the end

# each remaining race "arrives" in chronological order.
for race in pipeline_races[1:]:
    race_data = df[df['Race'] == race]
    X_race, y_race = get_features_target(race_data)

    # Drift check: how well does the deployed model predict this new race?
    preds = current_model.predict(X_race)
    mae = mean_absolute_error(y_race, preds)
    drifted = bool(mae > BASELINE_MAE * DRIFT_MULTIPLIER)

    entry = {
        'race': race,
        'current_model_mae_on_race': mae,
        'drift_detected': drifted,
        'deployed_version': registry['current_version'],
    }

    # The race is now known history, so add it to the training data
    # whether or not drift was detected
    training_data = pd.concat([training_data, race_data], ignore_index=True)
    trained_races.append(race)

    if drifted:
        # Retrain a candidate model on all history so far.
        candidate_model = train_model(training_data)

        # validation race that neither has trained on.
        current_val_mae = mean_absolute_error(y_val, current_model.predict(X_val))
        candidate_val_mae = mean_absolute_error(y_val, candidate_model.predict(X_val))

        entry['current_model_mae_on_validation'] = current_val_mae
        entry['candidate_mae_on_validation'] = candidate_val_mae

        version_counter += 1
        entry['candidate_version'] = version_counter

        if candidate_val_mae < current_val_mae:
            # Candidate is better: promote it to be the deployed model
            current_model = candidate_model
            entry['action'] = 'PROMOTED'
            registry['current_version'] = version_counter
        else:
            # Candidate is not better: current_model is left unchanged,
            entry['action'] = 'REJECTED' 

        # Save the candidate either way so every version is on record.
        entry['candidate_file'] = save_version(
            candidate_model, version_counter, trained_races, len(training_data),
            candidate_val_mae, entry['action'], race)
        save_registry()
    else:
        entry['action'] = 'NO ACTION (no drift)'

    log.append(entry)
    print(f"{race}: race_mae={mae:.3f}, drift={drifted}, "
          f"action={entry['action']}, deployed=v{registry['current_version']}")

# Summary 
# Turn the list of per-race entries into a table, print it and save it
log_df = pd.DataFrame(log)
print("\nFull log:")
print(log_df)
log_df.to_csv('data/pipeline_log.csv', index=False)
print("\nSaved log to data/pipeline_log.csv")
print(f"Saved model versions to models/ (registry: {REGISTRY_PATH})")
print(f"Final deployed model: v{registry['current_version']}")