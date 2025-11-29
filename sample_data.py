import numpy as np
import pandas as pd


def generate_sample_data(n_samples=1000):
    """
    Generate synthetic training data for the diet predictor model.

    Returns:
    X: DataFrame with dietary and lifestyle features
    y: Binary labels (0: healthy, 1: at risk)
    """
    np.random.seed(42)

    # Generate features
    sugar_intake = np.random.normal(50, 20, n_samples)
    salt_intake = np.random.normal(6, 2, n_samples)
    fat_intake = np.random.normal(70, 15, n_samples)
    sleep_hours = np.random.normal(7, 1.5, n_samples)
    steps_per_day = np.random.normal(7000, 2500, n_samples)
    water_intake = np.random.normal(2000, 500, n_samples)

    # Clip values to realistic ranges
    sugar_intake = np.clip(sugar_intake, 10, 150)
    salt_intake = np.clip(salt_intake, 2, 15)
    fat_intake = np.clip(fat_intake, 20, 150)
    sleep_hours = np.clip(sleep_hours, 3, 12)
    steps_per_day = np.clip(steps_per_day, 1000, 20000)
    water_intake = np.clip(water_intake, 500, 4000)

    # Create DataFrame
    X = pd.DataFrame({
        'sugar_intake_g': sugar_intake,
        'salt_intake_g': salt_intake,
        'fat_intake_g': fat_intake,
        'sleep_hours': sleep_hours,
        'steps_per_day': steps_per_day,
        'water_intake_ml': water_intake
    })

    # Generate labels based on risk factors
    risk_score = (
        (sugar_intake > 60) * 0.25 +
        (salt_intake > 7) * 0.25 +
        (fat_intake > 80) * 0.2 +
        (sleep_hours < 6) * 0.15 +
        (steps_per_day < 5000) * 0.1 +
        (water_intake < 1500) * 0.05
    )

    # Add some randomness
    risk_score += np.random.normal(0, 0.1, n_samples)

    # Convert to binary labels
    y = (risk_score > 0.5).astype(int)

    return X, y


if __name__ == "__main__":
    X, y = generate_sample_data()
    print("Sample data generated:")
    print(f"Features shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y)}")
    print("\nFirst few samples:")
    print(X.head())
