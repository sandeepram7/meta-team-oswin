import os
import json
import asyncio
import textwrap
from typing import List, Optional
from openai import OpenAI
from client import DataCurationEnvClient
from models import DataCleanAction

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

TASK_ID = os.getenv("TASK_ID", "task_1_breast_cancer")
MAX_STEPS = 5

# --- Logging Helpers ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- Agent Prompting ---
SYSTEM_PROMPT = """
You are a Senior Data Engineer. You receive statistics about a messy dataset and must choose ONE cleaning operation at a time to improve the data quality.
Your goal is to increase the downstream model's F1-score.
Available operations: 'identify_nan', 'classify_missingness', 'rm_outliers', 'impute_missing'.
Always specify the column you are working on.
"""

def get_agent_action(client: OpenAI, observation: str) -> DataCleanAction:
    """
    Calls the LLM to decide the next cleaning action.
    """
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
                    "description": "Apply a specific data cleaning operation.",
                    "parameters": DataCleanAction.model_json_schema()
                }
            }],
            tool_choice={"type": "function", "function": {"name": "clean_step"}}
        )
        
        args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return DataCleanAction(**args)
    except Exception as e:
        # Fallback action
        return DataCleanAction(operation="identify_nan", column="target", params={})

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Connect to the environment (assuming it's running locally for the baseline)
    async with DataCurationEnvClient(base_url="http://localhost:8000").async_iterator() as env:
        log_start(task=TASK_ID, env="DataCurationLab", model=MODEL_NAME)
        
        rewards = []
        steps_taken = 0
        final_score = 0.0
        success = False
        
        try:
            # 1. Reset
            result = await env.reset(task_id=TASK_ID)
            obs = result.observation
            
            for step in range(1, MAX_STEPS + 1):
                # Format observation for LLM
                obs_text = f"""
                Current Score: {obs.reward}
                Message: {obs.message}
                Missing %: {obs.missing_info}
                Dtypes: {obs.dtype_info}
                Data Head:
                {obs.df_head}
                """
                
                # 2. Get LLM Action
                action = get_agent_action(client, obs_text)
                
                # 3. Step Environment
                result = await env.step(action)
                obs = result.observation
                
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                final_score = (await env.state()).current_score
                
                log_step(step=step, action=f"{action.operation}({action.column})", 
                         reward=reward, done=result.done, error=None)
                
                if result.done:
                    success = final_score > 0.8 # Define a success threshold
                    break
            
            log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

        except Exception as e:
            print(f"[DEBUG] Error during inference: {e}")
            log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
