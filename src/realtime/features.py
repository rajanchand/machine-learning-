import math
import logging
from collections import deque
from typing import Dict, Any, List

logger = logging.getLogger("feature_extractor")

FEATURE_NAMES = [
    "duration",
    "log_orig_bytes",
    "log_resp_bytes",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "service_http",
    "service_dns",
    "service_ssh",
    "service_ssl",
    "service_other",
    "src_ip_conn_count_60s",
    "log_src_ip_orig_bytes_sum_60s"
]

class FeatureExtractor:
    def __init__(self, window_size_seconds: float = 60.0):
        self.window_size = window_size_seconds
        # Deque of tuples: (timestamp, src_ip, orig_bytes)
        self.history = deque()

    def _prune_history(self, current_ts: float):
        """Remove entries older than the window threshold."""
        cutoff = current_ts - self.window_size
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def process_flow(self, flow: Dict[str, Any], fit_mode: bool = False) -> List[float]:
        """
        Extract features from a single flow dict.
        Updates internal sliding window state unless fit_mode is True and we're doing static evaluation.
        (Usually fit_mode is just processing in sequence, so we update state).
        """
        ts = flow.get("ts", 0.0)
        src_ip = flow.get("src_ip", "")
        duration = float(flow.get("duration", 0.0))
        orig_bytes = float(flow.get("orig_bytes", 0.0))
        resp_bytes = float(flow.get("resp_bytes", 0.0))
        proto = flow.get("proto", "tcp").lower()
        service = flow.get("service", "other").lower()

        # Prune expired connections
        self._prune_history(ts)

        # Calculate stateful features BEFORE adding current flow to history
        # (This represents the state of the network just prior to/during this flow initiation)
        conn_count = 0
        bytes_sum = 0.0
        
        for hist_ts, hist_src_ip, hist_bytes in self.history:
            if hist_src_ip == src_ip:
                conn_count += 1
                bytes_sum += hist_bytes

        # Add current flow to history
        self.history.append((ts, src_ip, orig_bytes))

        # Log transform large byte fields to reduce skewness
        log_orig_bytes = math.log1p(orig_bytes)
        log_resp_bytes = math.log1p(resp_bytes)
        log_src_bytes_sum = math.log1p(bytes_sum)

        # One-hot encoding for proto
        proto_tcp = 1.0 if proto == "tcp" else 0.0
        proto_udp = 1.0 if proto == "udp" else 0.0
        proto_icmp = 1.0 if proto == "icmp" else 0.0

        # One-hot encoding for service
        service_http = 1.0 if service == "http" else 0.0
        service_dns = 1.0 if service == "dns" else 0.0
        service_ssh = 1.0 if service == "ssh" else 0.0
        service_ssl = 1.0 if service == "ssl" else 0.0
        service_other = 1.0 if service not in ["http", "dns", "ssh", "ssl"] else 0.0

        features = [
            duration,
            log_orig_bytes,
            log_resp_bytes,
            proto_tcp,
            proto_udp,
            proto_icmp,
            service_http,
            service_dns,
            service_ssh,
            service_ssl,
            service_other,
            float(conn_count),
            log_src_bytes_sum
        ]
        return features

    def reset(self):
        """Reset stateful window tracking."""
        self.history.clear()

    @staticmethod
    def get_feature_names() -> List[str]:
        return FEATURE_NAMES
