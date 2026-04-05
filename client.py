from typing import Dict, Any
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import DataCleanAction, DataCleanObservation, DataCleanState

class DataCurationEnvClient(EnvClient[DataCleanAction, DataCleanObservation, DataCleanState]):
    def _step_payload(self, action: DataCleanAction) -> dict:
        return {
            "operation": action.operation,
            "column": action.column,
            "params": action.params or {}
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=DataCleanObservation(
                done=payload.get("done", False),
                reward=payload.get("reward", 0.0),
                df_head=obs_data.get("df_head", ""),
                missing_info=obs_data.get("missing_info", {}),
                dtype_info=obs_data.get("dtype_info", {}),
                message=obs_data.get("message", ""),
            ),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> DataCleanState:
        return DataCleanState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            max_steps=payload.get("max_steps", 5),
            initial_score=payload.get("initial_score", 0.0),
            current_score=payload.get("current_score", 0.0),
        )
