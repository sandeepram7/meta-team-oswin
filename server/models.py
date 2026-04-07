from typing import List, Dict, Any, Optional
from openenv.core.env_server import Action, Observation, State

class DataCleanAction(Action):
    """The cleaning operation the agent wants to perform."""
    operation: str  # e.g., "impute_median", "impute_mode", "rm_outliers", "convert_dtype", "interpolate"
    column: str     # The name of the column to apply the operation to.
    params: Optional[Dict[str, Any]] = None # Additional parameters, e.g., {'target_dtype': 'float', 'alpha': 0.1}

class DataCleanObservation(Observation):
    """What the agent observes about the dataset."""
    # done and reward are inherited from Observation
    df_head: str                # Markdown representation of the first 5 rows.
    missing_info: Dict[str, float] # Percentage of missing values per column.
    dtype_info: Dict[str, str]    # Current data types of each column.
    message: str                 # Feedback from the last action.

class DataCleanState(State):
    """Internal state tracking for the environment."""
    # episode_id and step_count are inherited from State
    task_id: str
    max_steps: int
    initial_score: float
    current_score: float
