# Meta Data Curation Lab

An autonomous RL environment for systematic data curation and preprocessing. This project benchmark evaluates an agent's ability to maximize downstream model performance through a strategic, sequential data cleaning pipeline.

## Overview

The **Meta Data Curation Lab** presents a specialized OpenEnv environment where agents interact with degraded datasets. The agent's goal is to apply cleaning operations (imputation, outlier removal, encoding) to improve the F1-score or R2-score of a hidden downstream evaluator.

### Core Cleaning Pipeline

The environment enforces a strict 6-stage sequential execution model to ensure data integrity:

1. **`identify_nan`**: Normalizes common missing value placeholders.
2. **`drop_useless_columns`**: Prunes columns exceeding configurable missingness thresholds.
3. **`rm_outliers`**: Statistical outlier identification via univariate analysis.
4. **`impute_missing`**: Hybrid imputation (Median/Mode for MCAR, Random Forest for MAR).
5. **`encode_categoricals`**: Semantic cleanup followed by strategic One-Hot or Label encoding.
6. **`finish`**: Finalizes the dataset for the downstream evaluator.

---

## Architecture

- **`server/`**: The environment implementation.
  - `environment.py`: Orchestrates the RL logic, state backups, and auto-revert mechanics.
  - `preprocessing.py`: Core cleaning algorithms leveraging Scikit-Learn and custom logic.
  - `app.py`: FastAPI-based bridge between the environment and the OpenAI-compatible client.
- **`tasks/`**: Contains task-specific data and individual `grader.py` evaluation scripts.
- **`scripts/`**: Utility scripts for dataset ingestion and noise injection.
- **`inference.py`**: The autonomous agent implementation utilizing GPT-4o-mini.

---
