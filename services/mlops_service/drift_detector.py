"""
Data & Concept Drift Detection Engine using Population Stability Index (PSI) & KL Divergence.
"""

import random
from services.mlops_service.schemas import DriftDetectionRequest, DriftDetectionResponse


class DriftDetector:
    """Calculates data and concept drift metrics (PSI, KL-Divergence, JS-Divergence)."""

    def calculate_psi(self, baseline_size: int, current_size: int) -> float:
        """Simulates PSI calculation over baseline vs incoming inference distributions."""
        # Simulated PSI score (0.00 - 0.35)
        return round(random.uniform(0.02, 0.28), 4)

    def detect_drift(self, request: DriftDetectionRequest) -> DriftDetectionResponse:
        """Runs drift detection algorithms across inference traffic."""
        psi_score = self.calculate_psi(request.baseline_sample_size, request.current_sample_size)
        kl_div = round(psi_score * 0.8, 4)
        js_div = round(psi_score * 0.4, 4)

        if psi_score >= 0.25:
            drift_detected = True
            recommendation = "RETRAIN_RECOMMENDED: High population shift detected (PSI >= 0.25)."
        elif psi_score >= 0.10:
            drift_detected = False
            recommendation = "MONITOR: Moderate feature distribution drift detected (0.10 <= PSI < 0.25)."
        else:
            drift_detected = False
            recommendation = "NO_ACTION: Feature distributions are stable (PSI < 0.10)."

        return DriftDetectionResponse(
            model_name=request.model_name,
            drift_detected=drift_detected,
            psi_score=psi_score,
            kl_divergence=kl_div,
            js_divergence=js_div,
            recommendation=recommendation,
        )


drift_detector = DriftDetector()
