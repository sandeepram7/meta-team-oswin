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
from openenv.core.env_server import Environment
from models import DataCleanAction, DataCleanObservation, DataCleanState
from . import preprocessing

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
    MAX_STEPS = 5

    def __init__(self):
        self._state = None
        self._df = None
        self._target_col = None
        self._initial_score = 0.0
        self._current_score = 0.0
        self._miss_dict = {}

    def reset(self, seed=None, episode_id=None, **kwargs) -> DataCleanObservation:
        """
        Starts a new episode with a messy dataset.
        """
        task_id = kwargs.get("task_id", "task_1")
        # 1. Load Dataset (Mocking Task 1 if file not found)
        data_path = f"data/{task_id}_sampled.csv"
        if os.path.exists(data_path):
            self._df = pd.read_csv(data_path)
            # Ensure 'target' column exists as standardized by fetch_datasets.py
            if "target" not in self._df.columns:
                self._df["target"] = 0 # Fallback
        else:
            # Fallback to mock data if script wasn't run
            self._df = self._generate_mock_data(task_id)
        
        self._target_col = "target"
        self._miss_dict = {}
        
        # 2. Calculate Initial Baseline Score
        self._initial_score = self._calculate_quality_score()
        self._current_score = self._initial_score
        
        # 3. Setup State
        self._state = DataCleanState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            max_steps=self.MAX_STEPS,
            initial_score=self._initial_score,
            current_score=self._current_score
        )
        
        return self._get_obs(f"Dataset '{task_id}' loaded. Goal: Improve model F1-score.")

    def step(self, action: DataCleanAction, timeout_s=None, **kwargs) -> DataCleanObservation:
        """
        Executes one cleaning operation and returns the observation and reward.
        """
        self._state.step_count += 1
        op = action.operation.lower()
        col = action.column
        feedback = ""
        
        try:
            # --- Apply Preprocessing logic ---
            if op == "identify_nan":
                self._df = preprocessing.identify_NaN(self._df)
                feedback = f"Standardized dtypes and masked invalid values."
            
            elif op == "classify_missingness":
                self._miss_dict = preprocessing.classify_missingness(self._df)
                feedback = f"Classified missingness: {self._miss_dict}"
            
            elif op == "rm_outliers":
                alpha = action.params.get("alpha", 0.05) if action.params else 0.05
                self._df = preprocessing.rm_outlier_univar(self._df, alpha=alpha)
                feedback = f"Removed outliers from all numeric columns (alpha={alpha})."
            
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
                feedback = f"Applied hybrid imputation strategy."
            
            else:
                feedback = f"Unknown operation '{op}'. No changes made."

        except Exception as e:
            feedback = f"Action failed: {str(e)}"
            return self._get_obs(feedback, reward=-0.1)

        # --- Calculate Reward ---
        new_score = self._calculate_quality_score()
        reward = new_score - self._current_score # Delta improvement
        self._current_score = new_score
        self._state.current_score = new_score
        
        # --- Check Done Conditions ---
        # Done if we hit max steps or score is very high
        done = self._state.step_count >= self.MAX_STEPS or self._current_score > 0.95
        
        return self._get_obs(feedback, reward=reward, done=done)

    @property
    def state(self) -> DataCleanState:
        return self._state

    def _get_obs(self, message: str, reward: float = 0.0, done: bool = False) -> DataCleanObservation:
        """
        Generates the observation from the current DataFrame.
        """
        missing = (self._df.isna().mean() * 100).to_dict()
        dtypes = {col: str(dtype) for col, dtype in self._df.dtypes.items()}
        
        return DataCleanObservation(
            done=done,
            reward=reward,
            df_head=self._df.head(5).to_markdown(),
            missing_info=missing,
            dtype_info=dtypes,
            message=message
        )

    def _calculate_quality_score(self) -> float:
        """
        Calculates the predictive quality of the current dataframe.
        Auto-detects if it should use Regression (R2) or Classification (F1).
        """
        try:
            if self._df.empty or "target" not in self._df.columns:
                return 0.0
            
            # 1. Feature Selection: Use only numeric columns, drop target
            X = self._df.drop(columns=["target"])
            y = self._df["target"]
            
            # Ensure target is numeric for regression tasks
            if pd.api.types.is_object_dtype(y):
                y = pd.to_numeric(y, errors='coerce')
            
            X_numeric = X.select_dtypes(include=[np.number])
            
            if X_numeric.empty:
                return 0.1 
            
            # 2. Cleanup for Grader ONLY: Drop NaNs in target
            mask = ~y.isna()
            X_final = X_numeric[mask].copy()
            y_final = y[mask].copy()
            
            if len(y_final) < 10:
                return 0.0 # Not enough data after dropping NaNs
            
            # Simple internal imputation for features to make it trainable
            X_final = X_final.fillna(X_final.mean().fillna(0))
            
            # 3. Detect Task Type: If target has high cardinality, it's Regression
            unique_count = len(y_final.unique())
            is_regression = pd.api.types.is_numeric_dtype(y_final) and unique_count > 10
            
            if is_regression:
                model = DecisionTreeRegressor(max_depth=3, random_state=42)
                scores = cross_val_score(model, X_final, y_final, cv=3, scoring='r2')
                final_score = max(0.001, float(np.mean(scores))) # Ensure baseline > 0
            else:
                model = DecisionTreeClassifier(max_depth=3, random_state=42)
                scores = cross_val_score(model, X_final, y_final, cv=3, scoring='f1_macro')
                final_score = max(0.001, float(np.mean(scores)))
            
            return final_score

        except Exception:
            return 0.0

    def _generate_mock_data(self, task_id: str) -> pd.DataFrame:
        """
        Generates synthetic data for testing when CSV files are missing.
        """
        np.random.seed(42)
        n_rows = 500
        
        if "breast_cancer" in task_id or "task_1" in task_id:
            # Mock Task 1 (Mostly numeric, MCAR noise)
            data = {
                "radius": np.random.normal(15, 3, n_rows),
                "texture": np.random.normal(20, 5, n_rows),
                "perimeter": np.random.normal(100, 20, n_rows),
                "target": np.random.randint(0, 2, n_rows)
            }
            df = pd.DataFrame(data)
            # Inject noise
            for col in ["radius", "texture"]:
                mask = np.random.random(n_rows) < 0.1
                df.loc[mask, col] = np.nan
            return df
        else:
            # Default mock
            df = pd.DataFrame(np.random.randn(n_rows, 4), columns=['a', 'b', 'c', 'target'])
            df['target'] = (df['target'] > 0).astype(int)
            return df
