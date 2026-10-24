"""
train.py - Baseline model and evaluation.

Trains a RandomForest to predict lap time  on the season 2021 and tests it on later seasons. 
It reports MAE and RMSE against a naive baseline and saves three diagnostic plots to data/. 
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the cleaned dataset produced by main.py
df = pd.read_csv('data/processed_laps_multi.csv')

# Data sanity check
# Printing summary statistics and the slowest laps helped us find outlier
print("Lap time stats:")
print(df['LapTimeSeconds'].describe())
print("\nSlowest 10 laps in training data:")
print(df[df['Season']==2022].nlargest(10, 'LapTimeSeconds')[['Season','Circuit','Driver','LapNumber','LapTimeSeconds','TrackStatus']])

# encoding categorical columns as numbers.
df = pd.get_dummies(df, columns=['Driver', 'Team', 'Compound', 'Circuit'])

# Define target and features.
target = 'LapTimeSeconds'
drop_cols = ['LapTime', 'PitInTime', 'PitOutTime', target]  
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]     # inputs
y = df[target]           # correct answers

# Time-based split
# Train on the oldest season and test on later ones.
train_mask = df['Season'] == 2021
test_mask = df['Season'].isin([2022, 2023, 2024])

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

# Training the model
# 200 decision trees whose predictions are averaged.
# random_state=42 makes results reproducible.
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate on unseen later seasons 
predictions = model.predict(X_test)

# MAE: average miss in seconds.
# RMSE: like MAE but penalises large misses more heavily.
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print(f"MAE: {mae:.3f} seconds")
print(f"RMSE: {rmse:.3f} seconds")

# Naive baseline
# Always guess the average training lap time. The model must clearly beat
# this to show it has learned something real.
naive_pred = [y_train.mean()] * len(y_test) 
naive_mae = mean_absolute_error(y_test, naive_pred)
print(f"Naive baseline MAE (always predict average): {naive_mae:.3f} seconds")

# Plot 1: predicted vs actual lap time
# Each dot is one lap. The red dashed line is perfect prediction, so dots
# close to it mean accurate predictions.
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  
plt.xlabel('Actual Lap Time (s)')
plt.ylabel('Predicted Lap Time (s)')
plt.title('Predicted vs Actual Lap Time')
plt.tight_layout()
plt.savefig('data/predicted_vs_actual.png')
print("Saved plot: data/predicted_vs_actual.png")

# Prepare per-lap errors on the test set 
test_df = df[test_mask].copy()
test_df['Prediction'] = predictions
test_df['AbsError'] = (test_df['Prediction'] - test_df[target]).abs()

circuit_cols = [c for c in test_df.columns if c.startswith('Circuit_')]
test_df['CircuitName'] = test_df[circuit_cols].idxmax(axis=1).str.replace('Circuit_', '')

# Plot 2: error by circuit 
# Shows whether the model is equally accurate at every track.
error_by_circuit = test_df.groupby('CircuitName')['AbsError'].mean().sort_values()

plt.figure(figsize=(8, 5))
error_by_circuit.plot(kind='bar')
plt.ylabel('Mean Absolute Error (s)')
plt.title('Prediction Error by Circuit (Test Set)')
plt.tight_layout()
plt.savefig('data/error_by_circuit.png')
print("Saved plot: data/error_by_circuit.png")

# Plot 3: error by race
# Same idea, split by season and circuit. This shows how error changes
# over time (drift) and motivated the drift detection in pipeline.py.
test_df['Race'] = test_df['Season'].astype(str) + ' ' + test_df['CircuitName']
error_by_race = test_df.groupby('Race')['AbsError'].mean()

plt.figure(figsize=(8, 5))
error_by_race.plot(kind='bar', color='steelblue')
plt.ylabel('Mean Absolute Error (s)')
plt.title('Model Error by Race (Test Set, Chronological)')
plt.tight_layout()
plt.savefig('data/error_by_race.png')
print("Saved plot: data/error_by_race.png")

# plt.show() opens the figure windows and pauses the script until they
# are closed, so it is called once at the end for all three plots.
plt.show()

