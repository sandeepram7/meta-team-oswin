"""
Script to fetch and prepare datasets for the Meta Data Curation Lab benchmark.
"""

import os

import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

# Dataset configuration for the Data Curation Lab benchmarks
DATASETS = {
    "task_1": {
        "url": (
            "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main"
            "/sklearn/datasets/data/breast_cancer.csv"
        ),
        "target": "target",
        "name": "breast_cancer",
    },
    "task_2": {
        "url": (
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master"
            "/titanic.csv"
        ),
        "target": "Survived",
        "name": "titanic",
    },
    "task_3": {
        "url": (
            "https://raw.githubusercontent.com/STATCowboy/pbidataflowstalk/master"
            "/AmesHousing.csv"
        ),
        "target": "SalePrice",
        "name": "ames_housing",
    },
    "task_4": {
        "url": (
            "https://raw.githubusercontent.com/selva86/datasets/master" "/Cars93.csv"
        ),
        "target": "Price",
        "name": "cars93",
    },
    "task_5": {
        "url": (
            "https://raw.githubusercontent.com/MainakRepositor/Datasets-/master"
            "/zomato.csv"
        ),
        "target": "Aggregate rating",
        "name": "zomato",
    },
}


def inject_noise(df, p=0.25):
    """
    Apply dataset degradation for RL agent evaluation.
    1. Introduction of MCAR (Missing Completely At Random) values.
    2. Simulated outlier injection (extreme deviates).
    3. Row duplication to test deduplication logic.
    """
    df_noisy = df.copy()

    for col in df_noisy.columns:
        if (
            pd.api.types.is_numeric_dtype(df_noisy[col])
            and col != "target"
            and "id" not in col.lower()
        ):
            mask = np.random.rand(len(df_noisy)) < p
            df_noisy.loc[mask, col] = np.nan

            outlier_mask = np.random.rand(len(df_noisy)) < 0.05
            col_std = df_noisy[col].std()
            col_mean = df_noisy[col].mean()
            if not pd.isna(col_std):
                df_noisy.loc[outlier_mask, col] = col_mean + np.random.choice(
                    [-1, 1]
                ) * col_std * np.random.uniform(4, 6)

    n_dupes = int(len(df_noisy) * 0.05)
    if n_dupes > 0:
        dupes = df_noisy.sample(n=n_dupes, random_state=42)
        df_noisy = pd.concat([df_noisy, dupes], ignore_index=True)

    return df_noisy


def main():
    """Main execution loop for fetching and preparing datasets."""
    print("Preparing production datasets with advanced noise injection...")

    for task_id, info in DATASETS.items():
        print(f"Fetching {info['name']} for task: {task_id}")
        try:
            encoding = "ISO-8859-1" if info["name"] == "zomato" else "utf-8"
            df = pd.read_csv(info["url"], encoding=encoding)

            if info["target"] in df.columns and info["target"] != "target":
                df = df.rename(columns={info["target"]: "target"})

            sample_size = min(1000, len(df))
            df_sampled = df.sample(n=sample_size, random_state=42)

            df_noisy = inject_noise(df_sampled, p=0.25)

            output_path = f"data/{task_id}_sampled.csv"
            df_noisy.to_csv(output_path, index=False)
            print(f"Successfully saved to {output_path} ({len(df_noisy)} rows)")

        except Exception as e:
            print(f"Error fetching {info['name']}: {str(e)}")


if __name__ == "__main__":
    main()
