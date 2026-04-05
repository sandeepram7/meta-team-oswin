import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Dict, List, Optional, Union

def identify_NaN(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dtypes and mask invalid data as NaNs.
    Attempts to infer numeric types for string columns that look like numbers.
    """
    df = df.copy()
    for col in df.columns:
        # Try to convert to numeric, turning errors to NaN
        if df[col].dtype == 'object':
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                # If we got more than 50% numbers, keep the conversion
                if converted.notna().mean() > 0.5:
                    df[col] = converted
            except:
                pass
    return df

def classify_missingness(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify missingness for each column as MCAR, MAR, or MNAR.
    Simplified version using correlation thresholds.
    """
    missing_cols = df.columns[df.isna().any()].tolist()
    classifications = {}
    
    if not missing_cols:
        return classifications

    # Create indicator for missingness
    is_missing = df.isna()
    
    for col in missing_cols:
        # Check correlation of missingness in 'col' with values in other columns
        corrs = []
        for other_col in df.columns:
            if other_col == col: continue
            if df[other_col].dtype in ['float64', 'int64']:
                # Point-biserial correlation approximation
                # Correlation between is_missing[col] and df[other_col]
                corr = is_missing[col].corr(df[other_col])
                corrs.append(abs(corr))
        
        avg_corr = np.mean(corrs) if corrs else 0
        
        if avg_corr > 0.1:
            classifications[col] = "MAR" # Missing At Random (depends on other data)
        elif avg_corr < 0.05:
            classifications[col] = "MCAR" # Missing Completely At Random
        else:
            classifications[col] = "MNAR" # Missing Not At Random
            
    return classifications

def rm_outlier_univar(df: pd.DataFrame, alpha: float = 0.05, tail: str = "both", ind_col: List[str] = []) -> pd.DataFrame:
    """
    Remove univariate outliers using percentile trimming.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ind_col: continue # Skip ignored columns
        
        lower_q = df[col].quantile(alpha / 2 if tail == "both" else 0)
        upper_q = df[col].quantile(1 - alpha / 2 if tail == "both" else 1 - alpha)
        
        if tail == "both" or tail == "lower":
            df = df[df[col] >= lower_q]
        if tail == "both" or tail == "upper":
            df = df[df[col] <= upper_q]
            
    return df

def impute_mcar_mar(df: pd.DataFrame, miss_dict: Dict[str, str], 
                    random_impute: bool = False, 
                    nn_window: Optional[int] = None, 
                    force_missing_forest: bool = False,
                    n_trees: int = 10, tune: bool = False) -> pd.DataFrame:
    """
    Hybrid imputation strategy.
    MCAR -> Simple Median/Mode.
    MAR -> Iterative Random Forest (MissingForest).
    """
    df = df.copy()
    
    # 1. Simple Imputation for MCAR
    for col, m_type in miss_dict.items():
        if m_type == "MCAR" and not force_missing_forest:
            if df[col].dtype in [np.number]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)
                
    # 2. Iterative Imputation for Remaining (MAR/MNAR)
    # Simple MissingForest implementation
    remaining_missing = df.columns[df.isna().any()].tolist()
    if not remaining_missing:
        return df

    # Fill NaNs with median initially for features
    df_imputed = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.number]:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        else:
            df_imputed[col] = df_imputed[col].fillna("missing")

    for col in remaining_missing:
        # Prepare training data: rows where 'col' is not NaN
        train_idx = df[col].notna()
        test_idx = df[col].isna()
        
        if not test_idx.any(): continue
        
        X = df_imputed.drop(columns=[col])
        X = pd.get_dummies(X) # Simple encoding
        
        y = df.loc[train_idx, col]
        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        
        if df[col].dtype in [np.number]:
            model = RandomForestRegressor(n_estimators=n_trees, max_depth=5)
        else:
            model = RandomForestClassifier(n_estimators=n_trees, max_depth=5)
            
        model.fit(X_train, y)
        df_updated_vals = model.predict(X_test)
        df.loc[test_idx, col] = df_updated_vals
        
    return df
