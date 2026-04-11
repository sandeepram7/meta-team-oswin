import os
import uuid
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from openenv.core.env_server import Environment
from .models import DataCleanAction, DataCleanObservation, DataCleanState
from . import preprocessing

class DataCurationGrader:
    """Standalone Grader class for Phase 2 Evaluation."""
    def grade(self, state: DataCleanState) -> float:
        try:
            score = float(getattr(state, "current_score", 0.01))
            # Guard against NaN/Inf that bypasses exception handling
            if np.isnan(score) or np.isinf(score):
                return 0.01
            return max(0.01, min(score, 0.99))
        except Exception:
            return 0.01


class DataCurationEnv(Environment):
    """
    OpenEnv Environment for Data Curation (Cleaning and Preprocessing).
    The agent is rewarded based on how much its actions improve a downstream 
    Machine Learning model's performance.
    """
    ACTION_CLASS = DataCleanAction
    OBSERVATION_CLASS = DataCleanObservation
    STATE_CLASS = DataCleanState
    
    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 6  

    def __init__(self):
        self._state = None
        self._df = None
        self._target_col = None
        self._initial_score = 0.0
        self._current_score = 0.0
        self._miss_dict = {}

    def reset(self, seed=None, episode_id=None, **kwargs) -> DataCleanObservation:
        task_id = kwargs.get("task_id", "easy")
        
        # Map easy/medium/hard or task_n to internal task_1/2/3/4/5 numbering
        mapping = {
            "easy": "task_1", "medium": "task_2", "hard": "task_3",
            "task_4": "task_4", "task_5": "task_5"
        }
        internal_id = mapping.get(task_id, task_id)
        
        data_path = f"data/{internal_id}_sampled.csv"
        
        if os.path.exists(data_path):
            self._df = pd.read_csv(data_path)
            if "target" not in self._df.columns:
                self._df["target"] = 0 
        else:
            self._df = self._generate_mock_data(task_id)
        
        self._target_col = "target"
        self._miss_dict = {}
        
        self._initial_score = self._calculate_quality_score()
        self._current_score = self._initial_score
        
        self._state = DataCleanState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            max_steps=self.MAX_STEPS,
            initial_score=self._initial_score,
            current_score=self._current_score
        )
        
        return self._get_obs(f"Dataset '{task_id}' loaded. Goal: Improve model F1-score/R2.")

    def step(self, action: DataCleanAction, timeout_s=None, **kwargs) -> DataCleanObservation:
        self._state.step_count += 1
        
        op = action.operation.lower().strip()
        col = action.column
        feedback = ""
        
        # --- STATE BACKUP ---
        # Save a copy of the dataframe before we try to manipulate it
        prev_df = self._df.copy()
        prev_miss_dict = self._miss_dict.copy()
        prev_score = self._current_score
        # --------------------
        
        try:
            if op == "finish":
                feedback = "Agent successfully finished the pipeline."
                
            elif op == "identify_nan":
                self._df = preprocessing.identify_NaN(self._df)
                feedback = "[SUCCESS] Stage 1 complete. MUST move to Stage 2: drop_useless_columns."
            
            elif op == "drop_useless_columns":
                threshold = action.params.get("threshold", 50.0) if action.params else 50.0
                cols_before = set(self._df.columns)
                self._df = preprocessing.drop_useless_columns(self._df, threshold=float(threshold))
                dropped = cols_before - set(self._df.columns)
                feedback = f"[SUCCESS] Stage 2 complete. Dropped {len(dropped)} columns. MUST move to Stage 3: rm_outliers."
                
            elif op == "rm_outliers":
                alpha = action.params.get("alpha", 0.05) if action.params else 0.05
                self._df = preprocessing.rm_outlier_univar(self._df, alpha=float(alpha))
                feedback = "[SUCCESS] Stage 3 complete. MUST move to Stage 4: impute_missing."
                
            elif op == "impute_missing":
                if not self._miss_dict:
                    self._miss_dict = preprocessing.classify_missingness(self._df)
                params = action.params if action.params else {}
                self._df = preprocessing.impute_mcar_mar(
                    self._df, 
                    self._miss_dict,
                    random_impute=params.get("random_impute", False),
                    nn_window=params.get("nn_window", None),
                    force_missing_forest=params.get("force_missing_forest", False)
                )
                feedback = "[SUCCESS] Stage 4 complete. Data is imputed. MUST move to Stage 5: encode_categoricals."

            elif op == "encode_categoricals":
                self._df = preprocessing.encode_categoricals(self._df)
                feedback = "[SUCCESS] Stage 5 complete. Strings encoded. MUST move to Stage 6: finish."
            
            else:
                feedback = f"Unknown operation '{op}'. No changes made."

        except Exception as e:
            # If the code crashes (like the PyArrow error), revert the dataset!
            self._df = prev_df
            self._miss_dict = prev_miss_dict
            feedback = f"Action crashed: {str(e)}. [REVERTED] Dataset restored to previous safe state. MUST move to next stage."
            return self._get_obs(feedback, reward=0.0)

        # --- REWARD & REVERT LOGIC ---
        new_score = self._calculate_quality_score()
        actual_delta = new_score - self._current_score 
        
        if actual_delta < 0:
            # The action degraded the score! Revert it immediately.
            self._df = prev_df
            self._miss_dict = prev_miss_dict
            new_score = prev_score
            reward = 0.0 # Only sum positive rewards
            feedback += f" [REVERTED] Action degraded model by {actual_delta:.3f}. Dataset restored to previous safe state. MUST move to next stage."
        else:
            # Action was neutral or positive. Keep it!
            reward = actual_delta

        self._current_score = new_score
        self._state.current_score = new_score
        
        done = self._state.step_count >= self.MAX_STEPS or self._current_score > 0.95 or op == "finish"
        
        # CRITICAL: When the episode ends, report the cumulative quality score
        # as the final reward. The platform uses this as the "task score" and
        # requires it to be strictly between 0 and 1 (not 0.0, not 1.0).
        if done:
            reward = self._current_score
        
        # FINAL SAFETY CLAMP: Team Meta forbids exact 0.00 or 1.00
        # This ensures all rewards in logs are strictly within (0, 1)
        safe_reward = max(0.01, min(0.99, float(reward)))
        
        return self._get_obs(feedback, reward=safe_reward, done=done)


    @property
    def state(self) -> DataCleanState:
        return self._state

    def _get_obs(self, message: str, reward: float = 0.0, done: bool = False) -> DataCleanObservation:
        missing = (self._df.isna().mean() * 100).fillna(0.0).to_dict()
        dtypes = {col: str(dtype) for col, dtype in self._df.dtypes.items()}
        
        return DataCleanObservation(
            done=done,
            reward=reward,
            df_head=self._df.head(3).to_markdown() if not self._df.empty else "Empty DataFrame",
            missing_info=missing,
            dtype_info=dtypes,
            message=message
        )

    def _calculate_quality_score(self) -> float:
        try:
            if self._df.empty or "target" not in self._df.columns:
                return 0.01
            
            X = self._df.drop(columns=["target"])
            y = self._df["target"]
            
            mask = ~y.isna()
            X_final = X[mask].copy()
            y_final = y[mask].copy()
            
            if len(y_final) < 10 or X_final.empty:
                return 0.01 
            
            if y_final.dtype == 'object' or str(y_final.dtype) == 'category':
                is_regression = False
            else:
                unique_count = len(y_final.unique())
                is_regression = unique_count > 10
            
            numeric_cols = X_final.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X_final.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            
            numeric_transformer = SimpleImputer(strategy='mean')
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ], remainder='drop')
            
            if is_regression:
                clf = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', DecisionTreeRegressor(max_depth=5, random_state=42))
                ])
                scores = cross_val_score(clf, X_final, y_final, cv=3, scoring='r2')
                mean_score = float(np.nanmean(scores))
                if np.isnan(mean_score) or np.isinf(mean_score):
                    final_score = 0.01
                else:
                    final_score = max(0.01, min(mean_score, 0.99))
            else:
                clf = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
                ])
                scores = cross_val_score(clf, X_final, y_final, cv=3, scoring='f1_macro')
                mean_score = float(np.nanmean(scores))
                if np.isnan(mean_score) or np.isinf(mean_score):
                    final_score = 0.01
                else:
                    final_score = max(0.01, min(mean_score, 0.99))
            
            return final_score

        except Exception as e:
            return 0.01

    def _generate_mock_data(self, task_id: str) -> pd.DataFrame:
        np.random.seed(42)
        n_rows = 500
        
        if "breast_cancer" in task_id or "task_1" in task_id:
            data = {
                "radius": np.random.normal(15, 3, n_rows),
                "texture": np.random.normal(20, 5, n_rows),
                "perimeter": np.random.normal(100, 20, n_rows),
                "target": np.random.randint(0, 2, n_rows)
            }
            df = pd.DataFrame(data)
            for col in ["radius", "texture"]:
                mask = np.random.random(n_rows) < 0.1
                df.loc[mask, col] = np.nan
            return df
        else:
            df = pd.DataFrame(np.random.randn(n_rows, 4), columns=['a', 'b', 'c', 'target'])
            df['target'] = (df['target'] > 0).astype(int)
            return df