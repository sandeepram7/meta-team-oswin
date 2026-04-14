"""
Evaluation grader for the Meta Data Curation Lab benchmarks.
"""

import numpy as np


class DataCurationGrader:
    """
    Evaluation logic for the Data Curation Lab benchmarks.
    This class is discoverable by the OpenEnv benchmark suite for Phase 2 evaluation.
    Each task maintains a standalone grader to ensure robust discovery.
    """

    def grade(self, state) -> float:
        """
        Retrieves the final model quality score from the session state.
        Ensures the returned reward is bounded within (0.01, 0.99) as per platform requirements.
        """
        try:
            score = float(getattr(state, "current_score", 0.01))

            if not np.isfinite(score):
                return 0.01

            return max(0.01, min(score, 0.99))

        except Exception:
            return 0.01

