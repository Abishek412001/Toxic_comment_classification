"""
Moderation Policy Decision Engine evaluating probability thresholds to actions (PASS, FLAG, BLOCK).
"""

from typing import Dict, List, Tuple
from services.moderation_service.schemas import ToxicityScores, ActionEnum

# Default Threshold Config: (flag_threshold, block_threshold)
DEFAULT_CATEGORY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "toxic": (0.50, 0.85),
    "severe_toxic": (0.30, 0.60),
    "obscene": (0.50, 0.80),
    "threat": (0.25, 0.50),
    "insult": (0.50, 0.80),
    "identity_hate": (0.30, 0.60),
}


class ModerationPolicyEngine:
    """Evaluates multi-label toxicity scores against tenant policy rules."""

    def __init__(self, thresholds: Dict[str, Tuple[float, float]] = None):
        self.thresholds = thresholds or DEFAULT_CATEGORY_THRESHOLDS

    def evaluate(self, scores: ToxicityScores) -> Tuple[ActionEnum, bool, List[str], float]:
        """Evaluates scores and returns (action, flagged, flagged_categories, overall_risk_score)."""
        score_dict = scores.model_dump()
        flagged_categories: List[str] = []
        is_blocked = False
        is_flagged = False

        for category, score in score_dict.items():
            flag_thresh, block_thresh = self.thresholds.get(category, (0.50, 0.80))
            if score >= block_thresh:
                is_blocked = True
                flagged_categories.append(f"{category} (BLOCK)")
            elif score >= flag_thresh:
                is_flagged = True
                flagged_categories.append(f"{category} (FLAG)")

        max_risk_score = max(score_dict.values()) if score_dict else 0.0

        if is_blocked:
            action = ActionEnum.BLOCK
            flagged = True
        elif is_flagged:
            action = ActionEnum.FLAG
            flagged = True
        else:
            action = ActionEnum.PASS
            flagged = False

        return action, flagged, flagged_categories, max_risk_score
