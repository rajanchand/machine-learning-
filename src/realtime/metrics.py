from prometheus_client import Counter, Gauge, Histogram

# Ingestion Metrics
FLOWS_PROCESSED = Counter(
    "flows_processed_total",
    "Total number of network flow records consumed and processed",
    ["status"] # e.g. success, error
)

QUEUE_SIZE = Gauge(
    "ingestion_queue_size",
    "Current size of the ingestion queue"
)

# Inference Metrics
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Time taken to run feature extraction and Isolation Forest prediction",
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1]
)

ANOMALY_ALERTS = Counter(
    "anomaly_alerts_total",
    "Number of anomaly alerts emitted (after aggregation)",
    ["proto", "service"]
)

# Drift Metrics
DRIFT_P_VALUE = Gauge(
    "drift_p_value",
    "Kolmogorov-Smirnov p-value for anomaly score distribution (lower means drift)"
)

DRIFT_DETECTED = Gauge(
    "drift_detected",
    "Binary flag indicating whether concept drift is currently detected (1 = drifted, 0 = normal)"
)

# Analyst Feedback Metrics
ANALYST_FEEDBACK = Counter(
    "analyst_feedback_votes_total",
    "Total number of feedback votes submitted by analysts",
    ["vote"] # e.g. true_positive, false_positive
)
