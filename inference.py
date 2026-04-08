import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from client import DataCurationEnvClient
from server.models import DataCleanAction

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1") 
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini") 
API_KEY = os.getenv("OPENAI_API_KEY")

TASK_ID = os.getenv("TASK_ID", "task_1")
MAX_STEPS = 6 
MOCK_MODE = os.getenv("MOCK", "false").lower() == "true"

# --- Logging Helpers ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"\n[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}\n", flush=True)

# --- Agent Prompting ---
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

# --- Local Schema to Intercept Thoughts ---
class LLMToolSchema(BaseModel):
    thought: str = Field(..., description="Acknowledge the 'Last Action Executed' and the 'Environment Message'. State exactly why you are picking the NEXT logical operation in the sequence. NEVER repeat the last action, even if it was reverted.")
    operation: str = Field(..., description="The exact cleaning operation to apply. MUST be one of: 'identify_nan', 'drop_useless_columns', 'encode_categoricals', 'rm_outliers', 'impute_missing', 'finish'.")
    column: str = Field(..., description="The name of the column to target. Use 'all' for dataset-wide operations.")
    params: Optional[Dict[str, Any]] = Field(None, description="Optional parameters for the operation, e.g., {'alpha': 0.05} for rm_outliers or {'threshold': 50} for drop_useless_columns.")

def get_agent_action(client: Optional[OpenAI], observation: str) -> Tuple[str, DataCleanAction]:
    if MOCK_MODE:
        return "Mocking an action for testing.", DataCleanAction(operation="identify_nan", column="all", params={})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": observation}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "clean_step",
                    "description": "Apply a specific data cleaning operation to the environment.",
                    "parameters": LLMToolSchema.model_json_schema()
                }
            }],
            tool_choice={"type": "function", "function": {"name": "clean_step"}},
            temperature=0.1
        )
        
        args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        thought = args.pop("thought", "No thought provided.")
        
        return thought, DataCleanAction(**args)
    except Exception as e:
        print(f"[LLM ERROR] {type(e).__name__}: {e}")
        return f"Error occurred: {e}. Attempting fallback.", DataCleanAction(operation="finish", column="all", params={})

async def main():
    client = None
    if not MOCK_MODE:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    async with DataCurationEnvClient(base_url="http://localhost:8000") as env:
        log_start(task=TASK_ID, env="DataCurationLab", model=MODEL_NAME)
        
        rewards = []
        steps_taken = 0
        final_score = 0.0
        success = False
        
        try:
            result = await env.reset(task_id=TASK_ID)
            obs = result.observation
            reward = 0.0
            last_action_name = "None (Start of Episode)" 
            
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
                
                thought, action = get_agent_action(client, obs_text)
                print(f"\n[AGENT THOUGHT]: {thought}")
                
                last_action_name = action.operation
                
                result = await env.step(action)
                obs = result.observation
                
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                final_score = (await env.state()).current_score
                
                log_step(step=step, action=f"{action.operation}({action.column})", 
                         reward=reward, done=result.done, error=obs.message if "failed" in obs.message else None)
                
                if result.done:
                    # Update success to True if score > 0.60
                    success = round(final_score,1) >= 0.6
                    break
            
            log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

        except Exception as e:
            print(f"[DEBUG] Error during inference: {e}")
            log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())