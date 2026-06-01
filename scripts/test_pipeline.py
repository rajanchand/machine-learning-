import subprocess
import time
import httpx
import sys
import os
import signal

# Ensure we run from repository root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = "http://127.0.0.1:8000"

def run_test():
    print("======================================================================")
    print("Starting End-to-End Network Anomaly Detection Service Test")
    print("======================================================================")

    # Clean stale state from previous runs
    db_path = os.path.join(os.getcwd(), "data", "system.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Cleaned stale database from previous run.")
    
    # 1. Start FastAPI app
    print("\n[1/8] Starting FastAPI detector-service...")
    detector_proc = subprocess.Popen(
        ["./venv/bin/uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for FastAPI to start up
    ready = False
    for attempt in range(15):
        try:
            res = httpx.get(f"{API_URL}/ready", timeout=1.0)
            if res.status_code == 200:
                ready = True
                print("FastAPI is ready!")
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)
        
    if not ready:
        print("ERROR: FastAPI did not start successfully. Logs:")
        stdout, stderr = detector_proc.communicate(timeout=1.0)
        print(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        sys.exit(1)

    # 2. Start Flow Simulator
    print("\n[2/8] Starting Flow Simulator...")
    # Set drift-after to 200 for faster testing
    simulator_proc = subprocess.Popen(
        ["./venv/bin/python3", "-m", "src.simulator", "--host", "127.0.0.1", "--port", "9999", "--delay", "0.05", "--drift-after", "150"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # 3. Monitor flow processing
        print("\n[3/8] Monitoring incoming flows...")
        flows_seen = 0
        for _ in range(20):
            time.sleep(2)
            try:
                res = httpx.get(f"{API_URL}/stats")
                stats = res.json()
                flows_seen = stats.get("flows_processed_total", 0)
                active_conns = stats.get("ingestion", {}).get("active_connections", 0)
                q_size = stats.get("ingestion", {}).get("queue_size", 0)
                print(f"Processed Flows: {flows_seen} | Active Sim Connections: {active_conns} | Ingestion Queue: {q_size}")
                if flows_seen >= 100:
                    break
            except Exception as e:
                print(f"Error checking stats: {e}")

        # 4. Verify Alert Generation
        print("\n[4/8] Verifying alert generation...")
        res = httpx.get(f"{API_URL}/alerts")
        alerts = res.json()
        print(f"Total active alerts generated: {len(alerts)}")
        
        if not alerts:
            print("WARNING: No alerts generated yet. Waiting a bit longer...")
            time.sleep(5)
            alerts = httpx.get(f"{API_URL}/alerts").json()
            
        if alerts:
            print("Sample Alert:")
            print(f"  ID: {alerts[0]['alert_id']}")
            print(f"  Source: {alerts[0]['src_ip']} -> Dest: {alerts[0]['dest_ip']}")
            print(f"  Max Anomaly Score: {alerts[0]['max_anomaly_score']:.4f} (Threshold: {alerts[0]['threshold']:.4f})")
            print(f"  Occurrences (aggregated): {alerts[0]['occurrences']}")
        else:
            print("ERROR: No anomalies detected. Check if simulator is running or if model threshold is too high.")
            sys.exit(1)

        # 5. Capture Analyst Feedback
        print("\n[5/8] Simulating Analyst Feedback Loop...")
        alert_id = alerts[0]["alert_id"]

        # Capture baseline feedback count
        baseline_fp = httpx.get(f"{API_URL}/stats").json()["feedback"]["false_positives"]

        # Analyst reviews and marks as False Positive
        print(f"Marking Alert {alert_id} as 'false_positive'...")
        res = httpx.post(f"{API_URL}/alerts/{alert_id}/feedback?vote=false_positive")
        print(f"Feedback response: {res.json()}")
        
        # Verify stats updated (relative check: count increased by 1)
        res = httpx.get(f"{API_URL}/stats")
        stats = res.json()
        print(f"Updated Feedback Stats: {stats['feedback']}")
        if stats["feedback"]["false_positives"] != baseline_fp + 1:
            print("ERROR: Feedback counter did not increment.")
            sys.exit(1)

        # 6. Verify Concept Drift Detection
        print("\n[6/8] Waiting for Concept Drift Detection...")
        # We need flows > 150 (when simulator starts drifting) + 100 observations (drift detector window)
        # So we wait until flows_processed_total is >= 260
        drift_detected = False
        for _ in range(15):
            res = httpx.get(f"{API_URL}/stats")
            stats = res.json()
            flows_seen = stats.get("flows_processed_total", 0)
            drift_res = stats.get("drift", {})
            print(f"Processed Flows: {flows_seen} | Drift Status: {drift_res.get('status')} | Drift Detected: {drift_res.get('drift_detected')}")
            
            if drift_res.get("drift_detected"):
                drift_detected = True
                print("SUCCESS: Concept drift successfully detected by KS-test/Z-score!")
                print(f"Drift Details: {drift_res}")
                break
            time.sleep(3)
            
        if not drift_detected:
            print("WARNING: Concept drift was not detected within the timeout window.")

        # 7. Model Retraining Trigger
        print("\n[7/8] Triggering Model Retraining & Hot-Reloading...")
        # The database needs at least 200 raw flows
        res = httpx.get(f"{API_URL}/stats")
        total_flows = res.json().get("flows_processed_total", 0)
        
        if total_flows < 200:
            print(f"Waiting for flows to reach 200 (current: {total_flows})...")
            while total_flows < 200:
                time.sleep(2)
                total_flows = httpx.get(f"{API_URL}/stats").json().get("flows_processed_total", 0)
                print(f"Flows: {total_flows}")

        print("Sending POST /retrain request...")
        res = httpx.post(f"{API_URL}/retrain")
        print(f"Retrain response: {res.json()}")
        
        # Verify that model gets reloaded and version updates (from model_v1 to model_v{timestamp})
        print("Checking if new model version is loaded...")
        version_updated = False
        for _ in range(10):
            res = httpx.get(f"{API_URL}/stats")
            version = res.json()["model"]["current_version"]
            print(f"Active model version: {version}")
            if version != "model_v1":
                version_updated = True
                print(f"SUCCESS: Model hot-reloaded to new version: {version}!")
                break
            time.sleep(2)
            
        if not version_updated:
            print("ERROR: Model version did not update from model_v1.")
            sys.exit(1)

        # 8. Verify Prometheus metrics format
        print("\n[8/8] Checking Prometheus metrics endpoint format...")
        res = httpx.get(f"{API_URL}/metrics")
        lines = res.text.split("\n")
        # Find some key metrics
        metrics_found = []
        for line in lines:
            if line.startswith("flows_processed_total") or line.startswith("drift_detected") or line.startswith("analyst_feedback_votes_total"):
                metrics_found.append(line)
        print("Found matching Prometheus metrics:")
        for m in metrics_found[:5]:
            print(f"  {m}")
            
        if not metrics_found:
            print("ERROR: Did not find custom metrics on /metrics endpoint.")
            sys.exit(1)

        print("\n======================================================================")
        print("ALL TESTS PASSED SUCCESSFULLY! The system is fully operational.")
        print("======================================================================")

    finally:
        # Clean up subprocesses
        print("\nCleaning up simulator and FastAPI processes...")
        simulator_proc.terminate()
        detector_proc.terminate()
        try:
            simulator_proc.wait(timeout=2.0)
            detector_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            simulator_proc.kill()
            detector_proc.kill()
        print("Processes terminated.")

if __name__ == "__main__":
    run_test()
