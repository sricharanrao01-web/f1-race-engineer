""" main.py - Data aquisition and cleaning stage
 
    Downloads race lap data for selected circuits and seasons using the FastF1
    library, removes laps that are not representative of normal racing pace,
    and saves one combined table to data/processed_laps_multi.csv. 
"""
import fastf1
import pandas as pd

# Cache downloaded sessions in data/cache so that re-running the script
# loads them from disk instead of downloading from the F1 servers again.
fastf1.Cache.enable_cache('data/cache')

# function to load one race and return its cleaned lap data, or None if it fails.
def load_clean_race(year, circuit):
    # Returns a DataFrame with one row per valid lap, including the target column
    # LapTimeSeconds plus Season and Circuit labels.

    try:
        # 'R' only loads Race sessions but not 'Q' qualifying or 'FP1' practice.
        session = fastf1.get_session(year, circuit, 'R')
        session.load()
        laps = session.laps    # table with one row per driver per lap

        cols = ['Driver', 'Team', 'LapNumber', 'LapTime', 'Stint',
                'Compound', 'TyreLife', 'TrackStatus', 'IsAccurate',
                'PitInTime', 'PitOutTime']
        laps = laps[cols]

        # removing laps that do not reflect normal pace.
        laps = laps.dropna(subset=['LapTime'])
        laps = laps[laps['TrackStatus'] == '1']
        laps = laps[laps['IsAccurate'] == True]
        laps = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()]

        # Convert lap time from a Timedelta to plain seconds. This is the
        # target value the model learns to predict.
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

        # Labelling every lap with year and circuit.
        laps['Season'] = year
        laps['Circuit'] = circuit

        # removing the laps which are abnormally slow which passes the above filters.
        median_time = laps.groupby('Driver')['LapTimeSeconds'].transform('median')
        laps = laps[laps['LapTimeSeconds'] <= median_time + 10]

        return laps

    except Exception as e:
        # covers network problems, missing sessions, unexpected data, etc.
        print(f"Skipping {year} {circuit}: {e}")
        return None
# Races to download, as (season, circuit) pairs. Four seasons across three circuits.
races = [
    (2021, 'Bahrain'),
    (2021, 'Monza'),
    (2021, 'Silverstone'),
    (2022, 'Bahrain'),
    (2022, 'Monza'),
    (2022, 'Silverstone'),
    (2023, 'Bahrain'),
    (2023, 'Monza'),
    (2023, 'Silverstone'),
    (2024, 'Bahrain'),
    (2024, 'Monza'),
    (2024, 'Silverstone'),
]

# Load and clean each race, collecting the results in a list.
# Races that failed (returned None) are left out.
all_laps = []
for year, circuit in races:
    print(f"Loading {year} {circuit}...")
    data = load_clean_race(year, circuit)
    if data is not None:
        all_laps.append(data)

# Stack all per-race tables into one big table.
combined = pd.concat(all_laps, ignore_index=True)


print(f"Total rows across all races: {len(combined)}")
print(combined.head())     # preview of the first 5 rows.

# Save the cleaned dataset.
# train.py and pipeline.py read this file and use it as input.
combined.to_csv('data/processed_laps_multi.csv', index=False)
print("Saved to data/processed_laps_multi.csv")

