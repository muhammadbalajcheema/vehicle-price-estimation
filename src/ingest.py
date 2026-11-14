import os
import yaml
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/vehicles_raw.csv"

def ingest_data():
    os.makedirs("data/raw", exist_ok=True)
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    n = params["data"]["num_samples"]
    rng = np.random.default_rng(params["data"]["random_state"])

    brands = ["Toyota", "Honda", "Suzuki", "Kia", "Hyundai"]
    fuel_types = ["Petrol", "Hybrid", "Diesel"]
    transmissions = ["Automatic", "Manual"]
    assemblies = ["Local", "Imported"]

    brand_col = rng.choice(brands, size=n, p=[0.35, 0.28, 0.22, 0.08, 0.07])
    year_col = rng.integers(2010, 2026, size=n)
    age_col = 2026 - year_col
    mileage_col = np.clip(age_col * rng.integers(8000, 18000, size=n) + rng.integers(1000, 15000, size=n), 500, 300000)
    engine_cc_col = rng.choice([660, 1000, 1300, 1500, 1800, 2000], size=n, p=[0.18, 0.22, 0.25, 0.15, 0.15, 0.05])
    fuel_col = rng.choice(fuel_types, size=n, p=[0.72, 0.22, 0.06])
    trans_col = rng.choice(transmissions, size=n, p=[0.65, 0.35])
    assembly_col = rng.choice(assemblies, size=n, p=[0.75, 0.25])

    brand_premium = {"Toyota": 12.0, "Honda": 11.5, "Kia": 13.0, "Hyundai": 12.5, "Suzuki": 5.0}
    base_price = np.array([brand_premium[b] for b in brand_col])
    price = (
        base_price
        + (engine_cc_col / 100.0) * 1.6
        - (age_col * 1.45)
        - (mileage_col / 10000.0) * 0.35
        + np.where(fuel_col == "Hybrid", 6.5, 0.0)
        + np.where(trans_col == "Automatic", 3.2, 0.0)
        + np.where(assembly_col == "Imported", 4.0, 0.0)
        + rng.normal(0, 1.8, size=n)
    )
    price = np.round(np.clip(price, 6.5, 140.0), 2)

    df = pd.DataFrame({
        "brand": brand_col,
        "model_year": year_col,
        "mileage_km": mileage_col,
        "engine_cc": engine_col if False else engine_cc_col,
        "fuel_type": fuel_col,
        "transmission": trans_col,
        "assembly": assembly_col,
        "price_lakhs": price
    })

    df.loc[rng.choice(n, size=15, replace=False), "mileage_km"] = np.nan

    df.to_csv(RAW_PATH, index=False)
    print(f"Ingested {len(df)} vehicle listings into {RAW_PATH}")

if __name__ == "__main__":
    ingest_data()