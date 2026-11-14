import os
import pandas as pd

RAW_PATH = "data/raw/vehicles_raw.csv"
PROCESSED_PATH = "data/processed/vehicles_processed.csv"

def preprocess_data():
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(RAW_PATH)

    # Handle missing values and duplicates
    df["mileage_km"] = df["mileage_km"].fillna(df["mileage_km"].median())
    df = df.drop_duplicates().reset_index(drop=True)

    # Feature engineering
    df["vehicle_age"] = 2026 - df["model_year"]
    df["km_per_year"] = df["mileage_km"] / (df["vehicle_age"] + 1)

    # One-hot encode categorical variables
    categorical_cols = ["brand", "fuel_type", "transmission", "assembly"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    df_encoded.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed dataset saved to {PROCESSED_PATH} with shape {df_encoded.shape}")

if __name__ == "__main__":
    preprocess_data()