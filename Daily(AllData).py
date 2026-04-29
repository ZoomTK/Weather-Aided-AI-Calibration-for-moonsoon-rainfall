# -------------------------------------------------------------------------------
#  READY-TO-USE MACHINE LEARNING PIPELINE (Full Dataset Prediction)
#  Original version written by Brian Vant-Hull, City College of New York
#  Edited by Victor Carrion, Bronx Community College.
#
#  --- UPDATED ---
#  1. Trains on 80% of data.
#  2. Predicts on 100% of data (Train + Test) to show full performance.
#  3. Maintains Log Transform, Shuffle, and Deep Architecture.
# -------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras import layers, models, callbacks, regularizers
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------------
#  [ 1. USER CONFIGURATION BLOCK ]
# -------------------------------------------------------------------------------

# Path to your folder
DATA_FOLDER = r'C:\...\...\Downloads\PCCOE_PJ\gfs_imerg_site_precip_final\Try'

# Key Column Names
LABEL_COLUMN_NAME = 'daily_station_precip_mm'
IMERG_RAW_COL = 'nearest_px_precip'

# Output Files
BASELINE_PLOT_FILE = r'0_baseline_station_vs_imerg_FULL.png'
SCENARIO_1_PLOT = r'1_scenario_imerg_only_FULL.png'
SCENARIO_3_PLOT = r'3_scenario_full_weather_FULL.png'


# -------------------------------------------------------------------------------
#  [ 2. PLOTTING FUNCTIONS ]
# -------------------------------------------------------------------------------

def plot_comparison(x_data, y_data, title, filename):
    print(f"--- Generating Plot: {title} ---")

    mae = mean_absolute_error(x_data, y_data)
    r2 = r2_score(x_data, y_data)

    print(f"Metrics for {title} -> MAE: {mae:.4f}, R2: {r2:.4f}")

    limit = min(max(x_data.max(), y_data.max()), 150)
    plt.figure(figsize=(8, 8))

    sns.scatterplot(
        x=x_data,
        y=y_data,
        alpha=0.5,
        s=40,
        edgecolor='k',
        label=f'MAE: {mae:.4f}, $R^2$: {r2:.4f}'
    )

    plt.plot([0, limit], [0, limit], 'r--', lw=2, label="Perfect Line")

    plt.xlabel(f"Actual Station Precip ({LABEL_COLUMN_NAME})", fontsize=14)
    plt.ylabel("Predicted / Estimated Precip", fontsize=14)
    plt.title(f"{title}", fontsize=16)

    plt.xlim(0, limit)
    plt.ylim(0, limit)
    plt.legend(loc='upper left', fontsize='12')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename)
    print(f"--- Plot saved as '{filename}' ---\n")


# -------------------------------------------------------------------------------
#  [ 3. DATA LOADING & PROCESSING ]
# -------------------------------------------------------------------------------

def load_and_process_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not files:
        print("ERROR: No CSV files found.")
        return None

    data_frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            data_frames.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not data_frames: return None

    data = pd.concat(data_frames, ignore_index=True)

    if 'time_utc' in data.columns:
        data['time_utc'] = pd.to_datetime(data['time_utc'])
        data = data.sort_values(by='time_utc').reset_index(drop=True)
    else:
        print("ERROR: 'time_utc' column missing.")
        return None

    print("--- Feature Engineering: Lags disabled ---")

    data = data.dropna()
    print(f"--- Data Loaded. Rows: {len(data)} ---")

    return data


def get_feature_set(data, scenario, label_col):
    y = data[label_col]

    # Base columns
    numeric_cols = [
        'site_lat', 'site_lon',
        IMERG_RAW_COL, 'nearest_px_lat', 'nearest_px_lon'
    ]

    # Categorical columns
    categorical_cols = ['station_id']

    if scenario == 1:
        print("\n--- Config: Scenario 1 (IMERG Only) ---")
        features = numeric_cols

    elif scenario == 3:
        print("\n--- Config: Scenario 3 (Full Weather Data) ---")
        features = numeric_cols + [
            'cape_255', 'cin_255', 'pwat', 'interp_precip'
        ]

    valid_features = [c for c in features if c in data.columns]
    X = data[valid_features + categorical_cols].copy()
    numeric_features = [col for col in valid_features if col in X.columns]

    return X, y, numeric_features


# -------------------------------------------------------------------------------
#  [ 4. MODELING FUNCTIONS ]
# -------------------------------------------------------------------------------

def build_and_train(X_train, y_train):
    print("   -> Building Model Architecture...")
    model = models.Sequential()

    model.add(layers.Input(shape=(X_train.shape[1],)))

    # Relu + L2 + Dropout
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    callback = callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

    print("   -> Training Model (Max 200 Epochs)...")
    model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0, callbacks=[callback])

    return model


def run_experiment(data, scenario_id, plot_filename, plot_title):
    # 1. Prepare Data
    try:
        X, y, numeric_features = get_feature_set(data, scenario_id, LABEL_COLUMN_NAME)
    except Exception as e:
        print(f"Skipping Scenario {scenario_id}: {e}")
        return

    # --- LOG TRANSFORM STEP ---
    # 1. Transform Target (y)
    y_log = np.log1p(y)

    # 2. Transform Rain Features in X
    rain_cols = [IMERG_RAW_COL, 'interp_precip']
    for col in rain_cols:
        if col in X.columns:
            X[col] = np.log1p(np.maximum(X[col], 0))
    # --------------------------

    # 2. One-Hot Encode
    X_encoded = pd.get_dummies(X, columns=['station_id'])

    # 3. Split (Shuffle=True)
    dates = data['time_utc']
    # NOTE: We retain X_encoded and y_log (ALL DATA) for final prediction
    X_train, X_test, y_train_log, y_test_log, _, _ = train_test_split(
        X_encoded, y_log, dates, test_size=0.2, shuffle=True
    )

    # 4. Scale Numeric Features
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])

    # --- KEY CHANGE: Transform the FULL dataset for prediction ---
    # We use the scaler fitted on Train to scale ALL data
    X_full_scaled = X_encoded.copy()
    X_full_scaled[numeric_features] = scaler.transform(X_full_scaled[numeric_features])

    # 5. Train (on 80% Log Data)
    model = build_and_train(X_train, y_train_log)

    # 6. Predict on FULL DATASET (100% of Data)
    print("   -> Predicting on FULL DATASET...")
    y_pred_full_log = model.predict(X_full_scaled).flatten()

    # --- INVERSE TRANSFORM ---
    y_pred_full = np.expm1(y_pred_full_log)
    y_pred_full = np.maximum(y_pred_full, 0)  # Physics check

    # 7. Plot (Compare FULL Real vs Real)
    # Use 'y' (the original non-log target) for comparison
    plot_comparison(y, y_pred_full, f"{plot_title} (Full Dataset)", plot_filename)


# -------------------------------------------------------------------------------
#  [ 5. MAIN EXECUTION ]
# -------------------------------------------------------------------------------

def main():
    data = load_and_process_data(DATA_FOLDER)
    if data is None: return

    # Baseline (Full Data)
    plot_comparison(
        x_data=data[LABEL_COLUMN_NAME],
        y_data=data[IMERG_RAW_COL],
        title="Baseline: Raw IMERG vs. Station (Full Dataset)",
        filename=BASELINE_PLOT_FILE
    )

    # Scenario 1
    run_experiment(data, 1, SCENARIO_1_PLOT, "Scenario 1: IMERG Only")

    # Scenario 3
    run_experiment(data, 3, SCENARIO_3_PLOT, "Scenario 3: Full Weather Data")

    print("\n--- All Scenarios Complete. Check output folder. ---")


if __name__ == "__main__":
    main()