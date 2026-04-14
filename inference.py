"""
Inference script for the Meta Data Curation Lab benchmark.
This script acts as the agent, connecting to the OpenEnv server and executing data cleaning tasks using an LLM.
"""


import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from pydantic import BaseModel, Field

from server.models import DataCleanAction, DataCleanObservation, DataCleanState

load_dotenv()


class DataCurationEnvClient(
    EnvClient[DataCleanAction, DataCleanObservation, DataCleanState]
):
    """Client for connecting to the DataCurationEnv server."""

    def _step_payload(self, action: DataCleanAction) -> dict:
        return {
            "operation": action.operation,
            "column": action.column,
            "params": action.params or {},
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


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
MOCK_MODE = os.getenv("MOCK", "false").lower() == "true"

if not MOCK_MODE and HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

TASK_ID = os.getenv("TASK_ID", "task_1")
BENCHMARK = os.getenv("BENCHMARK", "DataCurationLab")
MAX_STEPS = 6


def log_start(task: str, env: str, model: str) -> None:
    """Logs the episode start according to Meta Hackathon rules."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    """Logs each step's result according to Meta Hackathon rules."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    safe_reward = max(0.01, min(0.99, float(reward)))
    print(
        f"[STEP]  step={step} action={action} reward={safe_reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Logs the episode completion according to Meta Hackathon rules."""
    rewards_str = ",".join(f"{max(0.01, min(0.99, r)):.2f}" for r in rewards)
    safe_score = max(0.01, min(0.99, float(score)))
    print(
        f"[END]   success={str(success).lower()} steps={steps} "
        f"score={safe_score:.2f} rewards={rewards_str}",
        flush=True,
    )


def extract_last_action_error(message: Optional[str]) -> Optional[str]:
    """Extracts critical errors from the environment's observation message."""
    if not message:
        return None

    lowered = message.lower()
    error_markers = ["failed", "crashed", "error", "unknown operation"]
    if any(marker in lowered for marker in error_markers):
        return message
    return None


SYSTEM_PROMPT = """You are an elite autonomous Data Engineer. 
Your goal is to maximize the downstream ML model's predictive score (F1-Score or R2).

CRITICAL PIPELINE RULES:
1. STRICT SEQUENTIAL EXECUTION: You MUST execute the following 6 stages in exact order. Look at the "Last Action Executed" in your context. YOU MUST MOVE TO THE NEXT STAGE. NEVER repeat the last action.
   - Stage 1: `identify_nan` 
   - Stage 2: `drop_useless_columns` 
   - Stage 3: `rm_outliers` (Run strictly on numbers before imputation)
   - Stage 4: `impute_missing` (Fills NaNs. Uses Classifiers for strings, Regressors for nums)
   - Stage 5: `encode_categoricals` (MUST run AFTER imputation so the imputer doesn't treat categories as continuous floats)
   - Stage 6: `finish`

2. AUTO-REVERT MECHANIC: If the environment message says "[REVERTED]", it means your last action dropped the score, and the dataset was automatically restored for you. DO NOT TRY TO FIX IT. Simply move on to the next stage in the sequence!

3. GLOBAL ACTIONS: Tools apply to the ENTIRE dataset at once. Use 'all' for the column argument.
"""


class LLMToolSchema(BaseModel):
    """Schema defining the structure expected from the LLM tool call."""

    thought: str = Field(
        ...,
        description="Acknowledge the 'Last Action Executed' and the 'Environment Message'. State exactly why you are picking the NEXT logical operation in the sequence. NEVER repeat the last action, even if it was reverted.",
    )
    operation: str = Field(
        ...,
        description="The exact cleaning operation to apply. MUST be one of: 'identify_nan', 'drop_useless_columns', 'encode_categoricals', 'rm_outliers', 'impute_missing', 'finish'.",
    )
    column: str = Field(
        ...,
        description="The name of the column to target. Use 'all' for dataset-wide operations.",
    )
    params: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional parameters for the operation, e.g., {'alpha': 0.05} for rm_outliers or {'threshold': 50} for drop_useless_columns.",
    )


def get_agent_action(
    client: Optional[OpenAI], observation: str
) -> Tuple[str, DataCleanAction]:
    """Queries the LLM for the next data cleaning action based on the observation."""
    if MOCK_MODE:
        return "Mocking an action for testing.", DataCleanAction(
            operation="identify_nan", column="all", params={}
        )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": observation},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "clean_step",
                        "description": "Apply a specific data cleaning operation to the environment.",
                        "parameters": LLMToolSchema.model_json_schema(),
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "clean_step"}},
            temperature=0.1,
        )

        args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        thought = args.pop("thought", "No thought provided.")

        return thought, DataCleanAction(**args)
    except Exception:
        return "LLM error; using safe fallback.", DataCleanAction(
            operation="finish", column="all", params={}
        )


async def run_single_task(
    client: Optional[OpenAI], env: DataCurationEnvClient, task_id: str
):
    """Executes a single data curation task from start to finish."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.01
    success = False
    crashed = False

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        reward = 0.0
        last_action_name = "None"

        for step in range(1, MAX_STEPS + 1):
            obs_text = f"""
            [ENVIRONMENT CONTEXT]
            Current Step: {step} out of {MAX_STEPS}
            Last Action Executed: {last_action_name}
            Reward for Last Action: {reward:.2f}
            Environment Message: {obs.message}

            [DATASET STATE]
            Missing %: {obs.missing_info}
            Dtypes: {obs.dtype_info}
            Data Head:
            {obs.df_head}
            """

            _, action = get_agent_action(client, obs_text)
            last_action_name = action.operation

            result = await env.step(action)
            obs = result.observation
            reward = float(result.reward or 0.01)
            done = bool(result.done)

            state = await env.state()
            final_score = float(state.current_score or 0.01)

            last_error = extract_last_action_error(obs.message)
            log_step(
                step=step,
                action=f"{action.operation}('{action.column}')",
                reward=reward,
                done=done,
                error=last_error,
            )

            rewards.append(reward)
            steps_taken = step
            if done:
                break

        success = bool(steps_taken > 0)
    except Exception:
        crashed = True
        success = False
    finally:
        log_end(
            success=success and not crashed,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )


async def main():
    """Main execution loop for running the benchmark tasks sequentially."""
    api_token = HF_TOKEN if not MOCK_MODE else "mock"
    api_client = (
        OpenAI(base_url=API_BASE_URL, api_key=api_token) if not MOCK_MODE else None
    )

    env_task = os.getenv("TASK_ID")
    tasks_to_run = (
        [env_task] if env_task else ["easy", "medium", "hard", "task_4", "task_5"]
    )

    env = DataCurationEnvClient(base_url="http://localhost:7860")
    async with env:
        for t_id in tasks_to_run:
            await run_single_task(api_client, env, t_id)


if __name__ == "__main__":
    asyncio.run(main())
