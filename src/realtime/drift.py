import numpy as np
import logging
from scipy.stats import ks_2samp
from typing import Dict, Any, List

logger = logging.getLogger("drift_detector")

class DriftDetector:
    def __init__(self, window_size: int = 200, p_value_threshold: float = 0.05, feature_z_threshold: float = 3.0):
        self.window_size = window_size
        self.p_value_threshold = p_value_threshold
        self.feature_z_threshold = feature_z_threshold
        
        # Sliding windows for serving-time monitoring
        self.observed_scores: List[float] = []
        self.observed_features: List[List[float]] = []

    def add_observation(self, features: List[float], anomaly_score: float):
        """Append a new inference observation to the sliding window."""
        self.observed_scores.append(anomaly_score)
        self.observed_features.append(features)

        # Enforce sliding window size
        if len(self.observed_scores) > self.window_size:
            self.observed_scores.pop(0)
            self.observed_features.pop(0)

    def check_drift(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for concept drift.
        Compares:
        1. Distribution of serving anomaly scores against baseline training scores (Kolmogorov-Smirnov test).
        2. Feature averages against baseline means using a standard Z-score test.
        """
        if len(self.observed_scores) < self.window_size:
            return {
                "drift_detected": False,
                "status": "insufficient_data",
                "observations_count": len(self.observed_scores),
                "window_size": self.window_size
            }

        baseline_scores = metadata.get("baseline_score_sample", [])
        if not baseline_scores:
            logger.warning("No baseline score sample found in model metadata. Cannot run KS drift test.")
            return {"drift_detected": False, "status": "missing_baseline"}

        # 1. Kolmogorov-Smirnov Test on anomaly scores
        # Null hypothesis (H0): Serving scores and baseline scores come from the same distribution.
        # If p-value < threshold, we reject H0 -> Drift detected.
        ks_stat, p_value = ks_2samp(self.observed_scores, baseline_scores)
        score_drift = bool(p_value < self.p_value_threshold)

        # 2. Z-score check on continuous features
        # We compare the mean of serving features to the training means/stds.
        ref_means = metadata.get("feature_means", [])
        ref_stds = metadata.get("feature_stds", [])
        feature_names = metadata.get("feature_names", [])
        
        drifted_features = []
        feature_drifts = {}
        
        if ref_means and ref_stds:
            obs_matrix = np.array(self.observed_features)
            obs_means = np.mean(obs_matrix, axis=0)
            
            for i, name in enumerate(feature_names):
                # Only check continuous features that have variance
                # (proto/service categories might have standard deviations of 0 if they never occurred in training)
                if i < len(ref_means) and i < len(ref_stds) and ref_stds[i] > 1e-5:
                    # Simple Z-statistic of the sample mean:
                    # z = (obs_mean - ref_mean) / (ref_std / sqrt(N))
                    z_stat = (obs_means[i] - ref_means[i]) / (ref_stds[i] / np.sqrt(self.window_size))
                    abs_z = abs(z_stat)
                    
                    feature_drifts[name] = {
                        "baseline_mean": ref_means[i],
                        "observed_mean": float(obs_means[i]),
                        "z_statistic": float(z_stat)
                    }
                    
                    if abs_z > self.feature_z_threshold:
                        drifted_features.append(name)
                        logger.warning(f"Feature '{name}' has drifted! Z-score: {z_stat:.4f} (Threshold: {self.feature_z_threshold})")

        drift_detected = score_drift or len(drifted_features) > 0

        result = {
            "drift_detected": drift_detected,
            "status": "calculated",
            "score_ks_statistic": float(ks_stat),
            "score_ks_p_value": float(p_value),
            "score_drift_detected": score_drift,
            "drifted_features": drifted_features,
            "feature_details": feature_drifts,
            "observations_count": len(self.observed_scores)
        }

        if drift_detected:
            logger.warning(f"CONCEPT DRIFT DETECTED! Anomaly score p-value: {p_value:.6f}. Labeled drifted features: {drifted_features}")
            
        return result

    def reset(self):
        self.observed_scores.clear()
        self.observed_features.clear()
        logger.info("Drift detector window reset.")
