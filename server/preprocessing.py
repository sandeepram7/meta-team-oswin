"""
Preprocessing module containing data curation algorithms and LLM-assisted cleaning steps.
"""


import json
import os

import numpy as np

try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd

from typing import Dict, List, Optional, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def identify_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Aggressively catch common placeholder strings and make them actual NaNs."""
    df = df.copy()
    df = df.replace(
        ["", " ", "NaN", "nan", "NA", "N/A", "null", "?", "-", "None"], np.nan
    )

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().mean() > 0.5:
                    df[col] = converted
            except Exception:
                pass
    return df


def drop_useless_columns(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """Prunes columns exceeding a configurable missingness threshold."""
    df = df.copy()
    missing_percentages = df.isna().mean() * 100

    cols_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
    if "target" in cols_to_drop:
        cols_to_drop.remove("target")

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def _llm_clean_categories(unique_vals: List[str]) -> Dict[str, str]:
    try:
        from openai import OpenAI

        api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("HF_TOKEN")
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

        if not api_key:
            return {}
        client = OpenAI(base_url=api_base_url, api_key=api_key)

        system_prompt = """You are a strict data cleaning assistant. 
        I will give you a list of unique categorical values. Identify duplicates caused by typos, whitespace, casing, or delimiters.
        Standardize them into clean, unified labels.
        Respond strictly with a valid JSON object where the key is the EXACT original string and the value is the standardized string.
        Example output: {"New York ": "New York", "new-york": "New York", "NY": "New York"}"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(unique_vals)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans messy strings using an LLM, then applies One-Hot Encoding (<5 unique)
    or Label Encoding (>=5 unique). Safely preserves NaNs for downstream imputation.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if "target" in cat_cols:
        cat_cols.remove("target")

    for col in cat_cols:
        unique_vals = df[col].dropna().unique().tolist()
        unique_vals = [str(v) for v in unique_vals]

        if 1 < len(unique_vals) <= 60:
            mapping = _llm_clean_categories(unique_vals)
            if mapping:
                df[col] = df[col].replace(mapping)
        else:
            df[col] = df[col].apply(
                lambda x: (
                    str(x).strip().title() if pd.notna(x) and isinstance(x, str) else x
                )
            )

        num_unique = df[col].nunique()
        mask = df[col].notna()

        if num_unique < 5:
            dummies = pd.get_dummies(df[[col]], columns=[col], dtype=float)

            if not mask.all():
                dummies.loc[~mask, :] = np.nan

            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            le = LabelEncoder()
            new_col = pd.Series(np.nan, index=df.index, dtype=float)

            if mask.any():
                new_col[mask] = le.fit_transform(df.loc[mask, col].astype(str))

            df[col] = new_col

    return df


def classify_missingness(df: pd.DataFrame) -> Dict[str, str]:
    """Classifies the type of missingness for columns with NaNs."""
    missing_cols = df.columns[df.isna().any()].tolist()
    classifications = {}
    if not missing_cols:
        return classifications

    is_missing = df.isna()
    for col in missing_cols:
        corrs = []
        for other_col in df.columns:
            if other_col == col:
                continue
            if df[other_col].dtype in ["float64", "int64"]:
                corr = is_missing[col].corr(df[other_col])
                if not np.isnan(corr):
                    corrs.append(abs(corr))
        avg_corr = np.nanmean(corrs) if corrs else 0.0
        if avg_corr > 0.1:
            classifications[col] = "MAR"
        elif avg_corr < 0.05:
            classifications[col] = "MCAR"
        else:
            classifications[col] = "MNAR"
    return classifications


def rm_outlier_univar(
    df: pd.DataFrame,
    alpha: float = 0.05,
    tail: str = "both",
    ind_col: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Removes statistical outliers via univariate quantile analysis."""
    if ind_col is None:
        ind_col = []
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ind_col or col == "target":
            continue
        lower_q = df[col].quantile(alpha / 2 if tail == "both" else 0)
        upper_q = df[col].quantile(1 - alpha / 2 if tail == "both" else 1 - alpha)

        if tail in ("both", "lower"):
            df.loc[df[col] < lower_q, col] = np.nan
        if tail in ("both", "upper"):
            df.loc[df[col] > upper_q, col] = np.nan
    return df


def impute_mcar_mar(
    df: pd.DataFrame,
    miss_dict: Dict[str, str],
    random_impute: bool = False,
    nn_window: Optional[int] = None,
    force_missing_forest: bool = False,
    n_trees: int = 10,
    tune: bool = False,
) -> pd.DataFrame:
    """Applies hybrid imputation: Median/Mode for MCAR, Random Forest for MAR/MNAR."""
    df = df.copy()

    for col, m_type in miss_dict.items():
        if m_type == "MCAR" and not force_missing_forest:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else np.nan
                )

    remaining_missing = df.columns[df.isna().any()].tolist()
    if not remaining_missing:
        return df

    df_imputed = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        else:
            df_imputed[col] = df_imputed[col].fillna("missing")

    for col in remaining_missing:
        train_idx = df[col].notna()
        test_idx = df[col].isna()

        if not test_idx.any() or not train_idx.any():
            continue

        x_feat = df_imputed.drop(columns=[col])
        x_feat = pd.get_dummies(x_feat)
        y = df.loc[train_idx, col]

        if len(y.unique()) == 1:
            df.loc[test_idx, col] = y.iloc[0]
            continue

        x_train = x_feat.loc[train_idx]
        x_test = x_feat.loc[test_idx]

        if pd.api.types.is_numeric_dtype(df[col]):
            model = RandomForestRegressor(n_estimators=n_trees, max_depth=5)
        else:
            model = RandomForestClassifier(n_estimators=n_trees, max_depth=5)

        model.fit(x_train, y)
        df.loc[test_idx, col] = model.predict(x_test)

    return df
