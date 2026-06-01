import asyncio
import os
import time
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.responses import RedirectResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pythonjsonlogger import jsonlogger

# Import our modular system components
from src.ingestion import IngestionServer
from src.features import FeatureExtractor
from src.model import ModelManager
from src.alerting import AlertingManager
from src.drift import DriftDetector
import src.metrics as metrics

# Setup Structured Logging as JSON
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log_handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.handlers = [log_handler]
root_logger.setLevel(logging.INFO)

logger = logging.getLogger("main_service")

# Global instances
ingestion_server = IngestionServer(host="0.0.0.0", port=9999, max_queue_size=100)
model_manager = ModelManager(models_dir="models", contamination=0.01)
alerting_manager = AlertingManager(db_path="data/system.db", suppression_window_seconds=120.0) # 2 mins for demo
drift_detector = DriftDetector(window_size=100, p_value_threshold=0.05) # 100 observations window
feature_extractor = FeatureExtractor(window_size_seconds=60.0)

consumer_task = None
flows_processed_count = 0
last_drift_result = {"drift_detected": False, "status": "no_data"}

async def process_flows_loop():
    """
    Core streaming consumer: extracts flows from ingestion queue,
    computes features, runs inference, stores results, and monitors drift.
    """
    global flows_processed_count, last_drift_result
    logger.info("Streaming consumer task started.")
    
    while True:
        try:
            flow = await ingestion_server.get_next_flow()
            metrics.QUEUE_SIZE.set(ingestion_server.queue.qsize())
            
            start_time = time.time()
            
            # 1. Feature Extraction
            features = feature_extractor.process_flow(flow)
            
            # 2. Model Prediction
            pred = model_manager.predict(features)
            score = pred["anomaly_score"]
            is_anomaly = pred["is_anomaly"]
            threshold = pred["threshold"]
            
            # Track latency
            metrics.INFERENCE_LATENCY.observe(time.time() - start_time)
            
            # 3. Store flow log with score
            alerting_manager.store_raw_flow(flow, score, is_anomaly)
            
            # 4. Handle Alerts and Aggregation
            if is_anomaly:
                alert = alerting_manager.add_potential_anomaly(flow, score, threshold)
                if alert:
                    # Increment Prometheus Counter for new/updated alerts
                    metrics.ANOMALY_ALERTS.labels(
                        proto=flow.get("proto", "tcp"),
                        service=flow.get("service", "other")
                    ).inc()
            
            # 5. Drift Monitoring
            drift_detector.add_observation(features, score)
            flows_processed_count += 1
            metrics.FLOWS_PROCESSED.labels(status="success").inc()
            
            # Check drift periodically (every 20 flows)
            if flows_processed_count % 20 == 0:
                meta = model_manager.metadata
                drift_res = drift_detector.check_drift(meta)
                last_drift_result = drift_res
                
                if drift_res.get("status") == "calculated":
                    metrics.DRIFT_P_VALUE.set(drift_res["score_ks_p_value"])
                    metrics.DRIFT_DETECTED.set(1.0 if drift_res["drift_detected"] else 0.0)
                    
        except asyncio.CancelledError:
            logger.info("Streaming consumer task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in streaming consumer loop: {e}", exc_info=True)
            metrics.FLOWS_PROCESSED.labels(status="error").inc()
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    global consumer_task
    logger.info("Initializing Network Anomaly Detection Service...")
    
    # Load model
    success = model_manager.load_model()
    if not success:
        logger.error("Failed to load a model. Bootstrapping a default model...")
        model_manager.bootstrap_initial_model()
        model_manager.load_model()

    # Start ingestion TCP server
    await ingestion_server.start()
    
    # Start consumer loop
    consumer_task = asyncio.create_task(process_flows_loop())
    
    yield
    
    # Shutdown sequence
    logger.info("Shutting down service...")
    if consumer_task:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
    await ingestion_server.stop()

app = FastAPI(
    title="Deployable Network Anomaly Detection Service",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/metrics")
def get_metrics():
    """Expose Prometheus metrics directly."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    """Verify system is up and the model is loaded."""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_version": model_manager.current_version,
        "active_connections": ingestion_server.active_connections,
        "flows_processed": flows_processed_count
    }

@app.get("/ready")
def ready_check():
    """Kubernetes-style readiness check."""
    if model_manager.model and ingestion_server.is_running:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/stats")
def get_stats():
    """Expose detailed system statistics (ingestion, drift, and feedback)."""
    return {
        "ingestion": ingestion_server.get_stats(),
        "model": {
            "current_version": model_manager.current_version,
            "threshold": model_manager.metadata.get("threshold"),
            "contamination": model_manager.metadata.get("contamination"),
            "training_samples": model_manager.metadata.get("num_training_samples"),
            "created_at": model_manager.metadata.get("created_at")
        },
        "feedback": alerting_manager.get_feedback_statistics(),
        "drift": last_drift_result,
        "flows_processed_total": flows_processed_count
    }

@app.get("/alerts")
def get_alerts(limit: int = 100):
    """Retrieve all alerts."""
    return alerting_manager.get_all_alerts(limit)

@app.get("/alerts/pending")
def get_pending_alerts():
    """Retrieve all un-reviewed (pending) alerts."""
    return alerting_manager.get_pending_alerts()

@app.post("/alerts/{alert_id}/feedback")
def submit_feedback(alert_id: str, vote: str):
    """
    Capture analyst feedback: 'true_positive' or 'false_positive'.
    Increments metrics and records it to SQLite for retraining.
    """
    if vote not in ["true_positive", "false_positive"]:
        raise HTTPException(status_code=400, detail="Vote must be 'true_positive' or 'false_positive'")
    
    success = alerting_manager.submit_feedback(alert_id, vote)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert with ID {alert_id} not found")
        
    metrics.ANALYST_FEEDBACK.labels(vote=vote).inc()
    return {"status": "success", "alert_id": alert_id, "vote": vote}

async def run_retraining_background():
    logger.info("Background retraining job started...")
    try:
        new_ver = model_manager.retrain_from_db("data/system.db")
        if new_ver:
            logger.info(f"Background retraining completed successfully. Loaded model: {new_ver}")
            # Reset drift detector to clear old scores
            drift_detector.reset()
        else:
            logger.warning("Background retraining did not yield a new model version.")
    except Exception as e:
        logger.error(f"Error in background retraining job: {e}", exc_info=True)

@app.post("/retrain")
def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Trigger model retraining on collected flows in SQLite.
    Runs asynchronously as a background task and hot-reloads the model.
    """
    # Verify we have enough flows before starting background task
    flows = alerting_manager.get_recent_raw_flows(limit=10)
    if len(flows) < 5: # Let's keep it small for testing, but in production it's 200+
        # Wait, our model retrain_from_db checks for 200 flows.
        # Let's check how many flows we actually have.
        recent_flows = alerting_manager.get_recent_raw_flows(limit=1000)
        if len(recent_flows) < 200:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient flows in database to retrain ({len(recent_flows)}/200 needed)."
            )

    background_tasks.add_task(run_retraining_background)
    return {"status": "retraining_triggered", "message": "Model retraining job scheduled in background."}
