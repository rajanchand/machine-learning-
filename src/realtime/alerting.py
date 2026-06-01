import sqlite3
import os
import time
import logging
import uuid
from typing import Dict, Any, List, Optional

logger = logging.getLogger("alerting_manager")

class AlertingManager:
    def __init__(self, db_path: str = "data/system.db", suppression_window_seconds: float = 300.0):
        self.db_path = db_path
        self.suppression_window = suppression_window_seconds
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialize the SQLite database schema if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    latest_timestamp REAL NOT NULL,
                    src_ip TEXT NOT NULL,
                    dest_ip TEXT NOT NULL,
                    proto TEXT NOT NULL,
                    service TEXT NOT NULL,
                    max_anomaly_score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    occurrences INTEGER NOT NULL DEFAULT 1,
                    feedback TEXT NOT NULL DEFAULT 'pending',
                    feedback_timestamp REAL
                )
            """)
            
            # Index for quick lookup during aggregation
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_lookup 
                ON alerts (src_ip, dest_ip, service, feedback, latest_timestamp)
            """)
            
            # Flow log history (raw flows) for retraining reference
            conn.execute("""
                CREATE TABLE IF NOT EXISTS flow_history (
                    uid TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    src_ip TEXT NOT NULL,
                    src_port INTEGER NOT NULL,
                    dest_ip TEXT NOT NULL,
                    dest_port INTEGER NOT NULL,
                    proto TEXT NOT NULL,
                    service TEXT NOT NULL,
                    duration REAL NOT NULL,
                    orig_bytes INTEGER NOT NULL,
                    resp_bytes INTEGER NOT NULL,
                    conn_state TEXT NOT NULL,
                    anomaly_score REAL,
                    is_anomaly INTEGER
                )
            """)
            conn.commit()
            logger.info("Database initialized successfully.")

    def store_raw_flow(self, flow: Dict[str, Any], score: Optional[float] = None, is_anomaly: Optional[bool] = None):
        """Store the raw flow log for historical archive and drift/retraining analysis."""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO flow_history 
                    (uid, ts, src_ip, src_port, dest_ip, dest_port, proto, service, duration, orig_bytes, resp_bytes, conn_state, anomaly_score, is_anomaly)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    flow.get("uid"),
                    flow.get("ts"),
                    flow.get("src_ip"),
                    flow.get("src_port"),
                    flow.get("dest_ip"),
                    flow.get("dest_port"),
                    flow.get("proto"),
                    flow.get("service"),
                    flow.get("duration"),
                    flow.get("orig_bytes"),
                    flow.get("resp_bytes"),
                    flow.get("conn_state"),
                    score,
                    1 if is_anomaly else 0
                ))
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to store raw flow: {e}")

    def add_potential_anomaly(self, flow: Dict[str, Any], score: float, threshold: float) -> Optional[Dict[str, Any]]:
        """
        Process a detected anomaly. Applies false-positive suppression/aggregation.
        Returns:
            dict representing the alert if a new or updated alert was emitted/updated.
        """
        src_ip = flow.get("src_ip", "")
        dest_ip = flow.get("dest_ip", "")
        proto = flow.get("proto", "tcp")
        service = flow.get("service", "other")
        now = time.time()
        
        with self._get_connection() as conn:
            # Check for an existing pending alert within the suppression window for this src->dest pair
            cutoff = now - self.suppression_window
            cursor = conn.execute("""
                SELECT * FROM alerts 
                WHERE src_ip = ? AND dest_ip = ? AND service = ? AND feedback = 'pending' AND latest_timestamp > ?
                LIMIT 1
            """, (src_ip, dest_ip, service, cutoff))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing alert (aggregate)
                alert_id = existing["alert_id"]
                new_occurrences = existing["occurrences"] + 1
                new_max_score = max(existing["max_anomaly_score"], score)
                
                conn.execute("""
                    UPDATE alerts 
                    SET latest_timestamp = ?, max_anomaly_score = ?, occurrences = ?
                    WHERE alert_id = ?
                """, (now, new_max_score, new_occurrences, alert_id))
                conn.commit()
                
                logger.info(f"Aggregated anomaly into alert {alert_id} (Occurrences: {new_occurrences})")
                return {
                    "alert_id": alert_id,
                    "timestamp": existing["timestamp"],
                    "latest_timestamp": now,
                    "src_ip": src_ip,
                    "dest_ip": dest_ip,
                    "proto": proto,
                    "service": service,
                    "max_anomaly_score": new_max_score,
                    "threshold": threshold,
                    "occurrences": new_occurrences,
                    "feedback": "pending"
                }
            else:
                # Create a new alert
                alert_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO alerts 
                    (alert_id, timestamp, latest_timestamp, src_ip, dest_ip, proto, service, max_anomaly_score, threshold, occurrences, feedback)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'pending')
                """, (alert_id, now, now, src_ip, dest_ip, proto, service, score, threshold))
                conn.commit()
                
                logger.info(f"Emitted NEW Alert! ID: {alert_id} | Src: {src_ip} -> Dest: {dest_ip} | Score: {score:.4f}")
                return {
                    "alert_id": alert_id,
                    "timestamp": now,
                    "latest_timestamp": now,
                    "src_ip": src_ip,
                    "dest_ip": dest_ip,
                    "proto": proto,
                    "service": service,
                    "max_anomaly_score": score,
                    "threshold": threshold,
                    "occurrences": 1,
                    "feedback": "pending"
                }

    def submit_feedback(self, alert_id: str, vote: str) -> bool:
        """
        Record analyst feedback (true_positive or false_positive) on an alert.
        """
        if vote not in ["true_positive", "false_positive", "pending"]:
            raise ValueError("Feedback vote must be 'true_positive', 'false_positive', or 'pending'")
            
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            alert = cursor.fetchone()
            if not alert:
                logger.error(f"Alert ID {alert_id} not found for feedback.")
                return False
                
            conn.execute("""
                UPDATE alerts 
                SET feedback = ?, feedback_timestamp = ?
                WHERE alert_id = ?
            """, (vote, time.time(), alert_id))
            conn.commit()
            
            logger.info(f"Captured feedback '{vote}' for Alert {alert_id}")
            return True

    def get_pending_alerts(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM alerts WHERE feedback = 'pending' ORDER BY latest_timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_all_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM alerts ORDER BY latest_timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_feedback_statistics(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            tp = conn.execute("SELECT COUNT(*) FROM alerts WHERE feedback = 'true_positive'").fetchone()[0]
            fp = conn.execute("SELECT COUNT(*) FROM alerts WHERE feedback = 'false_positive'").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM alerts WHERE feedback = 'pending'").fetchone()[0]
            return {"true_positives": tp, "false_positives": fp, "pending": pending}

    def get_labeled_flows_for_retraining(self) -> List[Dict[str, Any]]:
        """
        Get historical flows linked to alerts that have been labeled.
        This provides labeled data to help fine-tune or filter model predictions.
        """
        # Join alerts and flow_history on IPs to retrieve labeled training records.
        # This resolves the 'no labels in production' issue by extracting analyst feedback.
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT h.*, a.feedback 
                FROM flow_history h
                JOIN alerts a ON h.src_ip = a.src_ip AND h.dest_ip = a.dest_ip AND h.service = a.service
                WHERE a.feedback != 'pending'
            """)
            return [dict(row) for row in cursor.fetchall()]
            
    def get_recent_raw_flows(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve recent raw flows (for retraining baseline)."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM flow_history ORDER BY ts DESC LIMIT ?", (limit,))
            # Reverse to get chronological order
            flows = [dict(row) for row in cursor.fetchall()]
            flows.reverse()
            return flows
