import os
import pickle
import json
import time
import logging
from datetime import datetime
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from src.features import FeatureExtractor

logger = logging.getLogger("model_manager")

class ModelManager:
    def __init__(self, models_dir: str = "models", contamination: float = 0.01):
        self.models_dir = models_dir
        self.contamination = contamination
        self.model = None
        self.metadata = {}
        self.current_version = None

        # Ensure directory exists
        os.makedirs(self.models_dir, exist_ok=True)

    def get_latest_version(self) -> str:
        """Find the latest model version in the models directory."""
        files = [f for f in os.listdir(self.models_dir) if f.endswith(".bin")]
        if not files:
            return None
        
        # Sort files by name (assuming model_vX.bin format)
        # e.g., model_v1.bin, model_v2.bin
        versions = []
        for f in files:
            try:
                parts = f.replace(".bin", "").split("_v")
                if len(parts) == 2:
                    versions.append((int(parts[1]), f))
            except ValueError:
                continue
        
        if not versions:
            return None
        
        versions.sort()
        return versions[-1][1].replace(".bin", "")

    def load_model(self, version: str = "latest") -> bool:
        """Load a specific or the latest version of the model and its metadata."""
        if version == "latest":
            version = self.get_latest_version()
            if not version:
                logger.error("No model version found to load.")
                return False

        model_path = os.path.join(self.models_dir, f"{version}.bin")
        meta_path = os.path.join(self.models_dir, f"{version}_meta.json")

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            logger.error(f"Model or metadata file missing for version {version}")
            return False

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

            self.current_version = version
            logger.info(f"Successfully loaded model version {version}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {version}: {e}", exc_info=True)
            return False

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Predict if features are anomalous.
        Returns:
            dict containing:
                anomaly_score: float (higher is more anomalous, normalized roughly to [0,1])
                is_anomaly: bool
        """
        if not self.model:
            raise RuntimeError("Model is not loaded.")

        X = np.array(features).reshape(1, -1)
        
        # IsolationForest decision_function returns negative values for anomalies, positive for normal.
        # score_samples returns raw scores in range [-1, 0] approx, where lower (more negative) is anomaly.
        # We invert and shift so higher values represent anomalies.
        # Raw score is roughly -model.score_samples(X)[0]
        raw_score = -self.model.score_samples(X)[0] # value in [0, 1] approx
        
        threshold = self.metadata.get("threshold", 0.6)
        is_anomaly = raw_score > threshold

        return {
            "anomaly_score": float(raw_score),
            "is_anomaly": bool(is_anomaly),
            "threshold": threshold,
            "version": self.current_version
        }

    def train(self, flows: List[Dict[str, Any]], version: str) -> Tuple[str, str]:
        """
        Train a new Isolation Forest model on a list of flow dicts.
        Computes baseline feature distributions and threshold, saving model & metadata.
        """
        logger.info(f"Starting training for model version {version} on {len(flows)} flows...")
        
        # 1. Feature extraction in historical chronological order
        extractor = FeatureExtractor()
        X_list = []
        for flow in flows:
            feats = extractor.process_flow(flow)
            X_list.append(feats)
            
        X = np.array(X_list)
        
        # 2. Train model
        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X)

        # 3. Calculate baseline distribution and threshold
        # Compute baseline scores
        baseline_scores = -model.score_samples(X)
        
        # Threshold: set at the (1 - contamination) percentile
        # e.g. for contamination=0.01, threshold is at 99th percentile of normal training scores
        threshold = float(np.percentile(baseline_scores, (1 - self.contamination) * 100))
        
        # Compute feature means and standard deviations for drift detection
        feature_means = np.mean(X, axis=0).tolist()
        feature_stds = np.std(X, axis=0).tolist()
        
        # To support Kolmogorov-Smirnov test, we store a sample of baseline scores (e.g., up to 500 scores)
        # to compare against runtime serving distributions.
        baseline_score_sample = baseline_scores[:500].tolist()

        metadata = {
            "version": version,
            "timestamp": time.time(),
            "created_at": datetime.utcnow().isoformat(),
            "contamination": self.contamination,
            "threshold": threshold,
            "num_training_samples": len(flows),
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "baseline_score_sample": baseline_score_sample,
            "feature_names": FeatureExtractor.get_feature_names()
        }

        # 4. Save artifacts
        model_path = os.path.join(self.models_dir, f"{version}.bin")
        meta_path = os.path.join(self.models_dir, f"{version}_meta.json")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        with open(meta_path, "r+") if os.path.exists(meta_path) else open(meta_path, "w") as f:
            f.seek(0)
            json.dump(metadata, f, indent=2)
            f.truncate()

        logger.info(f"Model version {version} saved with threshold {threshold:.4f}")
        return model_path, meta_path

    def bootstrap_initial_model(self) -> str:
        """Create a basic v1 model using synthetic normal traffic to boot the system."""
        logger.info("Bootstrapping initial v1 model...")
        
        # Generate synthetic normal flows
        from src.simulator import FlowGenerator
        generator = FlowGenerator()
        
        # Generate 1200 normal flows (simulator starts with normal traffic)
        flows = []
        for _ in range(1200):
            # Bypass periodic anomalies for bootstrap baseline training
            flow = generator._generate_normal()
            # stagger timestamps slightly
            flow['ts'] = time.time() - (1200 - len(flows)) * 0.1
            flows.append(flow)
            
        model_path, _ = self.train(flows, "model_v1")
        return "model_v1"

    def retrain_from_db(self, db_path: str) -> Optional[str]:
        """Retrain the model using the historical flows stored in the database."""
        import sqlite3
        if not os.path.exists(db_path):
            logger.error(f"Database file {db_path} does not exist. Cannot retrain.")
            return None

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM flow_history ORDER BY ts ASC")
            rows = cursor.fetchall()
            conn.close()

            if len(rows) < 200:
                logger.warning(f"Insufficient historical flows ({len(rows)}/200 minimum) to retrain. Retraining aborted.")
                return None

            flows = [dict(row) for row in rows]
            new_version = f"model_v{int(time.time())}"
            self.train(flows, new_version)
            
            # Hot reload the model
            self.load_model(new_version)
            return new_version
        except Exception as e:
            logger.error(f"Error during retraining from database: {e}", exc_info=True)
            return None

if __name__ == "__main__":
    # Bootstrapping script
    logging.basicConfig(level=logging.INFO)
    manager = ModelManager()
    version = manager.bootstrap_initial_model()
    print(f"Bootstrapped model version: {version}")
