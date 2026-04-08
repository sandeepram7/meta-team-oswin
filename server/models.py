from typing import List, Dict, Any, Optional, Literal
from pydantic import Field
from openenv.core.env_server import Action, Observation, State

class DataCleanAction(Action):
    """The cleaning operation the agent wants to perform."""
    
    # Restrict operations to exactly these 6 commands
    operation: Literal[
        "identify_nan",
        "drop_useless_columns",
        "encode_categoricals",
        "rm_outliers",
        "impute_missing",
        "finish"
    ]
    column: str
    params: Optional[Dict[str, Any]] = None

class DataCleanObservation(Observation):
    """What the agent observes about the dataset."""
    df_head: str                
    missing_info: Dict[str, float] 
    dtype_info: Dict[str, str]    
    message: str                 

class DataCleanState(State):
    """Internal state tracking for the environment."""
    task_id: str
    max_steps: int
    initial_score: float
    current_score: float