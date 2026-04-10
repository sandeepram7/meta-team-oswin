import numpy as np

class DataCurationGrader:
    """Standalone Grader class for Phase 2 Evaluation.
    Relocated inside the 'tasks' package for official discovery."""
    
    def grade(self, state) -> float:
        try:
            # Safely extract current_score using getattr to handle any state object
            score = float(getattr(state, "current_score", 0.01))
            
            # Use numpy to check for non-finite values (NaN and Inf)
            if not np.isfinite(score):
                return 0.01
                
            # Clamp correctly within (0, 1) to pass Meta's strict range test
            return max(0.01, min(score, 0.99))
            
        except Exception:
            # Safety fallback: return a valid minimal score
            return 0.01
