import numpy as np

class DataCurationGrader:
    """Standalone Grader class for Phase 2 Evaluation.
    Isolated per task to ensure absolute discovery reliability."""
    
    def grade(self, state) -> float:
        try:
            # Safely extract score using getattr
            score = float(getattr(state, "current_score", 0.01))
            
            # Use numpy to check for non-finite values
            if not np.isfinite(score):
                return 0.01
                
            # Clamp correctly within (0, 1)
            return max(0.01, min(score, 0.99))
            
        except Exception:
            return 0.01
