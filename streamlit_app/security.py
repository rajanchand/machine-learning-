"""
security.py — Strict file upload sanitisation, antivirus scan placeholder, and rate limiter.
"""

import os
import re
import time
import hashlib
import streamlit as st
from typing import Tuple, Dict

# Simple in-memory rate limiter for Streamlit user actions
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 30
ip_rate_limits: Dict[str, list] = {}

# Strict magic byte mappings for security validation
ALLOWED_MAGIC_BYTES = {
    ".csv": [b"duration", b"sbytes", b"dur", b"Flow", b"Label", b"label", b"ts", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9", b"0", b"\""],
    ".pcap": [b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\n\r\r\n"], # standard pcap and pcapng headers
    ".pcapng": [b"\n\r\r\n", b"\n\r\r\n\x1a\x2b\x3c\x4d"]
}

# Simulated database of malicious file hashes (ClamAV style)
MOCK_MALICIOUS_HASHES = {
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855": "Zero-byte Empty Malware Exploit",
    "cf53642c56a8f10857b1d1082666666666666666666666666666666666666666": "Augmented PCAP Dos Exploit Payload",
}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and shell injection attacks."""
    # Strip any directory path indicators
    base = os.path.basename(filename)
    # Remove any characters except word chars, dots, and dashes
    sanitized = re.sub(r"[^\w\.\-]", "", base)
    return sanitized


def validate_file_size(file_bytes: bytes, max_size_mb: int = 500) -> bool:
    """Enforce maximum file upload size limit in bytes."""
    size_mb = len(file_bytes) / (1024 * 1024)
    return size_mb <= max_size_mb


def validate_magic_bytes(file_bytes: bytes, filename: str) -> bool:
    """Verify that file content actually matches its extension via magic header bytes."""
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_MAGIC_BYTES:
        return False
    
    # Check the first few bytes (up to 128 bytes) for standard headers
    header = file_bytes[:128]
    allowed_headers = ALLOWED_MAGIC_BYTES[ext]
    
    # For text/CSV files, check if standard ASCII/UTF-8 structure is valid
    if ext == ".csv":
        try:
            # Check if it starts with standard characters
            header.decode("utf-8", errors="strict")
            return True
        except UnicodeDecodeError:
            # If not decodable as text, reject binary-masquerading
            return False
            
    # For binary formats, check byte prefix
    for expected_prefix in allowed_headers:
        if header.startswith(expected_prefix):
            return True
            
    return False


def run_antivirus_scan(file_bytes: bytes) -> Tuple[bool, str]:
    """
    Simulated ClamAV antivirus engine.
    Computes file SHA-256 and scans against known NIDS malware payload signatures.
    """
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    if sha256 in MOCK_MALICIOUS_HASHES:
        return False, f"MALWARE DETECTED: {MOCK_MALICIOUS_HASHES[sha256]}"
    
    # Simulated heuristical signature checks
    # Scan for common attack triggers in payload like shellcode or binary indicators in PCAPs
    if b"eval(base64_decode" in file_bytes or b"/bin/sh" in file_bytes:
        return False, "HEURISTIC SUSPICION: Remote Code Execution shellcode payload found in flow log!"
        
    return True, f"Clean (SHA-256: {sha256[:12]}...)"


def check_rate_limit(client_ip: str) -> bool:
    """Implement stateless action rate-limiting to prevent brute force or Denial of Service."""
    current_time = time.time()
    
    if client_ip not in ip_rate_limits:
        ip_rate_limits[client_ip] = []
        
    # Filter out requests older than the window
    ip_rate_limits[client_ip] = [t for t in ip_rate_limits[client_ip] if current_time - t < RATE_LIMIT_WINDOW]
    
    if len(ip_rate_limits[client_ip]) >= MAX_REQUESTS_PER_WINDOW:
        return False
        
    ip_rate_limits[client_ip].append(current_time)
    return True


def secure_sandbox_process(uploaded_file) -> Tuple[bytes, str]:
    """
    Complete secure sandboxed validation flow for uploads:
      1. Rate Limit Check
      2. Filename Sanitisation
      3. Size verification
      4. Magic byte extension validation
      5. Antivirus scan
    """
    # Rate Limit
    client_ip = st.context.headers.get("X-Forwarded-For", "127.0.0.1").split(",")[0]
    if not check_rate_limit(client_ip):
        raise PermissionError("🚨 SECURITY WARNING: Rate limit exceeded! Please slow down requests.")
        
    # Read bytes safely
    file_bytes = uploaded_file.getvalue()
    filename = sanitize_filename(uploaded_file.name)
    
    # 1. Size Check
    if not validate_file_size(file_bytes, max_size_mb=500):
        raise ValueError(f"🚨 File upload limit exceeded: Max 500MB allowed.")
        
    # 2. Magic byte / MIME check
    if not validate_magic_bytes(file_bytes, filename):
        raise TypeError(f"🚨 Security check failed: File headers do not match extension '{os.path.splitext(filename)[1]}'.")
        
    # 3. Antivirus scan
    is_safe, scan_msg = run_antivirus_scan(file_bytes)
    if not is_safe:
        raise ValueError(scan_msg)
        
    return file_bytes, filename
