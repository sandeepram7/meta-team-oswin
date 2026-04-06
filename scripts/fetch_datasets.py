import os
import pandas as pd
import numpy as np

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

DATASETS = {
    "task_1": {
        "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/breast_cancer.csv",
        "target": "target",
        "name": "breast_cancer"
    },
    "task_2": {
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "target": "Survived",
        "name": "titanic"
    },
    "task_3": {
        "url": "https://raw.githubusercontent.com/STATCowboy/pbidataflowstalk/master/AmesHousing.csv",
        "target": "SalePrice",
        "name": "ames_housing"
    },
    "task_4": {
        "url": "https://raw.githubusercontent.com/selva86/datasets/master/Cars93.csv",
        "target": "Price",
        "name": "cars93"
    },
    "task_5": {
        "url": "https://raw.githubusercontent.com/MainakRepositor/Datasets-/master/zomato.csv",
        "target": "Aggregate rating",
        "name": "zomato"
    }
}

def inject_noise(df, p=0.25):
    """
    Advanced Noise Injection (as per ai-ans.md recommendations):
    1. MCAR (Missing Completely At Random) NaNs.
    2. Outlier injection (extreme values).
    3. Duplicate row injection.
    """
    df_noisy = df.copy()
    
    # 1. MCAR: Random NaNs
    for col in df_noisy.columns:
        if pd.api.types.is_numeric_dtype(df_noisy[col]) and col != "target":
            mask = np.random.rand(len(df_noisy)) < p
            df_noisy.loc[mask, col] = np.nan
            
            # 2. Outlier injection: ~5% of rows get extreme values
            outlier_mask = np.random.rand(len(df_noisy)) < 0.05
            col_std = df_noisy[col].std()
            col_mean = df_noisy[col].mean()
            if not pd.isna(col_std):
                df_noisy.loc[outlier_mask, col] = col_mean + np.random.choice([-1, 1]) * col_std * np.random.uniform(4, 6)

    # 3. Duplicate rows: inject ~5% duplicate rows
    n_dupes = int(len(df_noisy) * 0.05)
    if n_dupes > 0:
        dupes = df_noisy.sample(n=n_dupes, random_state=42)
        df_noisy = pd.concat([df_noisy, dupes], ignore_index=True)
        
    return df_noisy

def main():
    print("🚀 Starting Production Data Preparation (Advanced Noise)...")
    
    for task_id, info in DATASETS.items():
        print(f"📥 Fetching {info['name']} for {task_id}...")
        try:
            # Handle encoding for Zomato
            encoding = 'ISO-8859-1' if info['name'] == 'zomato' else 'utf-8'
            df = pd.read_csv(info['url'], encoding=encoding)
            
            # Standardize target
            if info['target'] in df.columns and info['target'] != "target":
                df = df.rename(columns={info['target']: "target"})
            
            # Sample to 1000 rows
            sample_size = min(1000, len(df))
            df_sampled = df.sample(n=sample_size, random_state=42)
            
            # Inject advanced noise
            df_final = inject_noise(df_sampled, p=0.25)
            
            # Save
            output_path = f"data/{task_id}_sampled.csv"
            df_final.to_csv(output_path, index=False)
            print(f"✅ Saved to {output_path} ({len(df_final)} rows)")
            
        except Exception as e:
            print(f"❌ Failed to fetch {info['name']}: {e}")

if __name__ == "__main__":
    main()
