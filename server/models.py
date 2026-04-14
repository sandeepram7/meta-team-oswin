"""
Data models for the Data Curation Lab environment.
"""

from typing import Any, Dict, Literal, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class DataCleanAction(Action):
    """The cleaning operation the agent wants to perform."""

    operation: Literal[
        "identify_nan",
        "drop_useless_columns",
        "encode_categoricals",
        "rm_outliers",
        "impute_missing",
        "finish",
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
