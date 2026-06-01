import asyncio
import json
import random
import time
import argparse
import sys
import logging
from datetime import datetime

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": "%(message)s"}'
)
logger = logging.getLogger("flow_simulator")

# Normal IP lists
NORMAL_SOURCES = [f"192.168.1.{i}" for i in range(10, 100)]
NORMAL_DESTINATIONS = [
    "10.0.0.1", "10.0.0.2", "10.0.0.3", # Internal servers
    "8.8.8.8", "1.1.1.1",              # Public DNS
    "192.168.1.1"                      # Gateway
]
SERVICES = ["http", "dns", "ssh", "ssl", "other"]
PROTOCOLS = ["tcp", "udp", "icmp"]

class FlowGenerator:
    def __init__(self, start_drift_after: int = 500):
        self.flow_count = 0
        self.start_drift_after = start_drift_after
        self.drift_active = False

    def generate_flow(self) -> dict:
        self.flow_count += 1
        
        # Check if we should trigger drift
        if self.flow_count > self.start_drift_after and not self.drift_active:
            self.drift_active = True
            logger.info(f"Concept drift triggered! Shifted baseline distribution after {self.start_drift_after} flows.")

        # Determine traffic type: Anomaly, Drifted, or Normal
        # 1. Periodic Anomaly (every 50 flows)
        if self.flow_count % 50 == 0:
            anomaly_type = random.choice(["exfiltration", "port_scan", "dos"])
            return self._generate_anomaly(anomaly_type)
        
        # 2. Drifted Traffic
        if self.drift_active:
            return self._generate_drifted()
        
        # 3. Normal Traffic
        return self._generate_normal()

    def _generate_normal(self) -> dict:
        src = random.choice(NORMAL_SOURCES)
        dest = random.choice(NORMAL_DESTINATIONS)
        proto = random.choices(PROTOCOLS, weights=[0.7, 0.25, 0.05])[0]
        
        if proto == "udp":
            service = "dns"
            duration = random.uniform(0.01, 0.1)
            orig_bytes = random.randint(40, 120)
            resp_bytes = random.randint(60, 300)
            dest_port = 53
        elif proto == "icmp":
            service = "other"
            duration = 0.0
            orig_bytes = 64
            resp_bytes = 64
            dest_port = 0
        else: # tcp
            service = random.choices(SERVICES, weights=[0.5, 0.0, 0.1, 0.3, 0.1])[0]
            if service == "http":
                dest_port = 80
                duration = random.uniform(0.1, 2.0)
                orig_bytes = random.randint(150, 1000)
                resp_bytes = random.randint(500, 10000)
            elif service == "ssl":
                dest_port = 443
                duration = random.uniform(0.5, 10.0)
                orig_bytes = random.randint(1000, 20000)
                resp_bytes = random.randint(2000, 100000)
            elif service == "ssh":
                dest_port = 22
                duration = random.uniform(2.0, 60.0)
                orig_bytes = random.randint(2000, 50000)
                resp_bytes = random.randint(2000, 80000)
            else:
                dest_port = random.choice([8080, 5000, 3000])
                duration = random.uniform(0.05, 1.0)
                orig_bytes = random.randint(100, 500)
                resp_bytes = random.randint(100, 500)

        return {
            "ts": time.time(),
            "uid": f"C{self.flow_count:06d}",
            "src_ip": src,
            "src_port": random.randint(1024, 65535),
            "dest_ip": dest,
            "dest_port": dest_port,
            "proto": proto,
            "service": service,
            "duration": duration,
            "orig_bytes": orig_bytes,
            "resp_bytes": resp_bytes,
            "conn_state": "SF"
        }

    def _generate_drifted(self) -> dict:
        # In drifted state, users download significantly more data (e.g. video streaming trend starts)
        # and connections take longer. This simulates a shift in feature distributions.
        src = random.choice(NORMAL_SOURCES)
        dest = random.choice(NORMAL_DESTINATIONS)
        proto = "tcp"
        service = "ssl"
        dest_port = 443
        
        # Drift: higher average duration and resp_bytes
        duration = random.uniform(10.0, 45.0)  # normally 0.5-10
        orig_bytes = random.randint(5000, 30000)
        resp_bytes = random.randint(200000, 800000) # normally 2k-100k
        
        return {
            "ts": time.time(),
            "uid": f"C{self.flow_count:06d}",
            "src_ip": src,
            "src_port": random.randint(1024, 65535),
            "dest_ip": dest,
            "dest_port": dest_port,
            "proto": proto,
            "service": service,
            "duration": duration,
            "orig_bytes": orig_bytes,
            "resp_bytes": resp_bytes,
            "conn_state": "SF"
        }

    def _generate_anomaly(self, anomaly_type: str) -> dict:
        logger.info(f"Injecting anomaly of type: {anomaly_type} (Flow #{self.flow_count})")
        attacker_ip = "192.168.1.200"
        
        if anomaly_type == "exfiltration":
            # Huge data transfer from a single local host to an external IP
            return {
                "ts": time.time(),
                "uid": f"C{self.flow_count:06d}",
                "src_ip": attacker_ip,
                "src_port": 49152,
                "dest_ip": "203.0.113.50", # Attacker command & control / storage
                "dest_port": 443,
                "proto": "tcp",
                "service": "ssl",
                "duration": 300.5,
                "orig_bytes": 850000000, # 850 MB!
                "resp_bytes": 12000,
                "conn_state": "SF"
            }
        elif anomaly_type == "port_scan":
            # Rapid connections to multiple destination ports, low duration/bytes
            target_ip = "10.0.0.1"
            return {
                "ts": time.time(),
                "uid": f"C{self.flow_count:06d}",
                "src_ip": attacker_ip,
                "src_port": random.randint(30000, 60000),
                "dest_ip": target_ip,
                "dest_port": random.choice([21, 22, 23, 25, 80, 110, 443, 445, 3389]),
                "proto": "tcp",
                "service": "other",
                "duration": 0.01,
                "orig_bytes": 0,
                "resp_bytes": 0,
                "conn_state": "S0" # Connection attempt, no reply
            }
        else: # dos
            # Rapid flood of UDP packets
            return {
                "ts": time.time(),
                "uid": f"C{self.flow_count:06d}",
                "src_ip": attacker_ip,
                "src_port": 53,
                "dest_ip": "10.0.0.2",
                "dest_port": 80,
                "proto": "udp",
                "service": "other",
                "duration": 0.001,
                "orig_bytes": 1400,
                "resp_bytes": 0,
                "conn_state": "S0"
            }

async def stream_flows(host: str, port: int, delay: float, start_drift_after: int):
    generator = FlowGenerator(start_drift_after=start_drift_after)
    
    while True:
        try:
            logger.info(f"Connecting to ingestion receiver at {host}:{port}...")
            reader, writer = await asyncio.open_connection(host, port)
            logger.info("Connected successfully! Starting stream...")
            
            while True:
                flow = generator.generate_flow()
                payload = json.dumps(flow) + "\n"
                
                writer.write(payload.encode())
                await writer.drain() # Crucial for backpressure! If receiver's socket is blocked, drain() will yield and block here.
                
                await asyncio.sleep(delay)
                
        except (ConnectionRefusedError, OSError) as e:
            logger.warning(f"Connection failed: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error in stream: {e}", exc_info=True)
            await asyncio.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Network Flow Simulator")
    parser.add_argument("--host", default="127.0.0.1", help="Ingestion server host")
    parser.add_argument("--port", type=int, default=9999, help="Ingestion server port")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between flows in seconds")
    parser.add_argument("--drift-after", type=int, default=500, help="Trigger drift after this many flows")
    args = parser.parse_args()

    try:
        asyncio.run(stream_flows(args.host, args.port, args.delay, args.drift_after))
    except KeyboardInterrupt:
        logger.info("Simulator stopped.")
