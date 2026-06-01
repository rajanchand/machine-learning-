"""
utils.py — Logging, model save/load, session helpers, and timing utilities.

This module provides shared infrastructure used across the entire pipeline:
  - Structured logging configuration
  - Model persistence (joblib for sklearn/xgboost, torch.save for PyTorch)
  - Reproducibility seeding
  - Timer decorator for profiling
"""

import os
import time
import logging
import random
import functools
import joblib
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger("nids")

# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")

# ──────────────────────────────────────────────────────────────────────
# Device detection
# ──────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    logger.info(f"Using device: {dev}")
    return dev

# ──────────────────────────────────────────────────────────────────────
# Model persistence
# ──────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved")
os.makedirs(MODELS_DIR, exist_ok=True)

import hashlib

class ModelSignatureException(Exception):
    """Custom exception raised when model signature verification fails."""
    pass


def calculate_sha256(filepath: str) -> str:
    """Calculate SHA-256 hash of a file securely."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def sign_model_file(filepath: str):
    """Generate and write a secure SHA-256 signature sidecar file."""
    sha = calculate_sha256(filepath)
    sig_path = filepath + ".sha256"
    with open(sig_path, "w") as f:
        f.write(sha)
    logger.info(f"🛡️ Model signed securely: {sig_path} (Hash: {sha[:10]}...)")


def verify_model_file(filepath: str) -> bool:
    """Verify that a model matches its signature file. Raises exception on mismatch."""
    sig_path = filepath + ".sha256"
    if not os.path.exists(sig_path):
        logger.warning(f"⚠️ Model signature missing: {sig_path}")
        return False
    
    current_sha = calculate_sha256(filepath)
    with open(sig_path, "r") as f:
        stored_sha = f.read().strip()
        
    if current_sha != stored_sha:
        logger.critical(f"🚨 SECURITY VIOLATION: Model hash mismatch! Tampering detected: {filepath}")
        return False
        
    logger.info(f"🛡️ Model verified securely: {filepath}")
    return True


def security_audit_log(event_type: str, user: str, status: str, details: str, severity: str = "INFO"):
    """Write structured security audit records for OWASP standards."""
    log_msg = f"[AUDIT] Event: {event_type} | User: {user} | Status: {status} | Details: {details}"
    if severity == "CRITICAL":
        logger.critical(log_msg)
    elif severity == "WARNING":
        logger.warning(log_msg)
    else:
        logger.info(log_msg)


def save_sklearn_model(model, name: str, metadata: dict = None):
    """Save a scikit-learn / XGBoost model with optional metadata and sign it securely."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    payload = {"model": model, "metadata": metadata or {}}
    joblib.dump(payload, path)
    logger.info(f"Saved sklearn model → {path}")
    sign_model_file(path)
    return path


def load_sklearn_model(name: str):
    """Load and verify a scikit-learn / XGBoost model's integrity."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    # Secure Signature Verification
    if not verify_model_file(path):
        raise ModelSignatureException(f"Security Check Failed: Model '{name}' has been tampered with or lacks a signature.")
        
    payload = joblib.load(path)
    logger.info(f"Loaded sklearn model ← {path}")
    return payload["model"], payload.get("metadata", {})


def save_torch_model(model, name: str, metadata: dict = None):
    """Save a PyTorch model state dict with metadata and sign it securely."""
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, path)
    logger.info(f"Saved PyTorch model → {path}")
    sign_model_file(path)
    return path


def load_torch_model(model_class, name: str, **model_kwargs):
    """Load and verify a PyTorch model's integrity."""
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    # Secure Signature Verification
    if not verify_model_file(path):
        raise ModelSignatureException(f"Security Check Failed: PyTorch Model '{name}' has been tampered with or lacks a signature.")
        
    checkpoint = torch.load(path, map_location="cpu")
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info(f"Loaded PyTorch model ← {path}")
    return model, checkpoint.get("metadata", {})


def save_preprocessing_artifacts(scaler, label_encoder_map, feature_names, name="preprocessing"):
    """Save preprocessing artifacts (scaler, encoders, feature list)."""
    path = os.path.join(MODELS_DIR, f"{name}_artifacts.joblib")
    joblib.dump({
        "scaler": scaler,
        "label_encoders": label_encoder_map,
        "feature_names": feature_names,
    }, path)
    logger.info(f"Saved preprocessing artifacts → {path}")
    return path


def load_preprocessing_artifacts(name="preprocessing"):
    """Load preprocessing artifacts."""
    path = os.path.join(MODELS_DIR, f"{name}_artifacts.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessing artifacts not found: {path}")
    artifacts = joblib.load(path)
    logger.info(f"Loaded preprocessing artifacts ← {path}")
    return artifacts

# ──────────────────────────────────────────────────────────────────────
# List saved models
# ──────────────────────────────────────────────────────────────────────

def list_saved_models():
    """List all saved model files in the models directory."""
    models = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith((".joblib", ".pt")):
            path = os.path.join(MODELS_DIR, f)
            models.append({
                "name": f,
                "path": path,
                "size_mb": round(os.path.getsize(path) / 1e6, 2),
                "modified": time.ctime(os.path.getmtime(path)),
            })
    return models

# ──────────────────────────────────────────────────────────────────────
# Timer decorator
# ──────────────────────────────────────────────────────────────────────

def timer(func):
    """Decorator that logs execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper
