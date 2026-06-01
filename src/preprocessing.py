"""
preprocessing.py — Data loading, cleaning, scaling, encoding, and splitting.

Supports:
  - CIC-IDS2017 CSV files (auto-detected by column names like 'Flow Duration')
  - UNSW-NB15 CSV files (auto-detected by column names like 'sbytes', 'dbytes')
  - Generic CSV with a 'Label' or 'label' column
  - PCAP files (basic flow feature extraction via scapy)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils import get_logger, timer, save_preprocessing_artifacts

logger = get_logger("preprocessing")

# ──────────────────────────────────────────────────────────────────────
# Dataset format detection
# ──────────────────────────────────────────────────────────────────────

CIC_IDS_MARKERS = {"Flow Duration", "Total Fwd Packets", "Fwd Packet Length Max"}
UNSW_NB15_MARKERS = {"sbytes", "dbytes", "sttl", "dttl", "attack_cat"}


def detect_dataset_format(df: pd.DataFrame) -> str:
    """Detect if a DataFrame is CIC-IDS2017, UNSW-NB15, or generic."""
    cols = set(df.columns.str.strip())
    if CIC_IDS_MARKERS.issubset(cols):
        return "CIC-IDS2017"
    if UNSW_NB15_MARKERS.issubset(cols):
        return "UNSW-NB15"
    return "generic"

# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

@timer
def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV dataset, strip column whitespace, and log shape."""
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    logger.info(f"Loaded {filepath}: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def load_pcap(filepath: str) -> pd.DataFrame:
    """
    Extract basic flow-level features from a PCAP file.
    This is a simplified extractor — for production use CICFlowMeter or Zeek.
    """
    try:
        from scapy.all import rdpcap, IP, TCP, UDP
    except ImportError:
        raise ImportError("scapy is required for PCAP parsing. Install with: pip install scapy")

    packets = rdpcap(filepath)
    flows = []
    for pkt in packets:
        if IP in pkt:
            row = {
                "src_ip": pkt[IP].src,
                "dst_ip": pkt[IP].dst,
                "protocol": pkt[IP].proto,
                "length": len(pkt),
                "ttl": pkt[IP].ttl,
            }
            if TCP in pkt:
                row["src_port"] = pkt[TCP].sport
                row["dst_port"] = pkt[TCP].dport
                row["flags"] = str(pkt[TCP].flags)
            elif UDP in pkt:
                row["src_port"] = pkt[UDP].sport
                row["dst_port"] = pkt[UDP].dport
                row["flags"] = ""
            else:
                row["src_port"] = 0
                row["dst_port"] = 0
                row["flags"] = ""
            flows.append(row)

    df = pd.DataFrame(flows)
    logger.info(f"Extracted {len(df)} packets from PCAP: {filepath}")
    return df

# ──────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────

@timer
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw dataset:
      - Replace inf with NaN, then drop NaN rows
      - Drop constant (zero-variance) columns
      - Drop duplicate rows
    """
    initial = len(df)

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN
    df.dropna(inplace=True)
    logger.info(f"Dropped {initial - len(df)} rows with NaN/inf values")

    # Drop constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        logger.info(f"Dropped {len(constant_cols)} constant columns: {constant_cols[:5]}...")

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    df.reset_index(drop=True, inplace=True)
    return df

# ──────────────────────────────────────────────────────────────────────
# Label extraction
# ──────────────────────────────────────────────────────────────────────

def extract_labels(df: pd.DataFrame, dataset_format: str):
    """
    Extract binary labels (0 = normal, 1 = attack) from the dataset.
    Returns (df_without_label, labels_series, label_column_name).
    """
    if dataset_format == "CIC-IDS2017":
        label_col = "Label" if "Label" in df.columns else "label"
        if label_col not in df.columns:
            raise ValueError("CIC-IDS2017 dataset must contain a 'Label' column")
        labels = (df[label_col].str.strip().str.upper() != "BENIGN").astype(int)
        df_clean = df.drop(columns=[label_col])
        return df_clean, labels, label_col

    elif dataset_format == "UNSW-NB15":
        label_col = "label" if "label" in df.columns else "Label"
        attack_cat_col = "attack_cat" if "attack_cat" in df.columns else None
        cols_to_drop = [label_col]
        if attack_cat_col and attack_cat_col in df.columns:
            cols_to_drop.append(attack_cat_col)
        labels = df[label_col].astype(int)
        df_clean = df.drop(columns=cols_to_drop)
        return df_clean, labels, label_col

    else:
        # Generic: look for 'Label', 'label', 'class', 'target'
        for col in ["Label", "label", "class", "target", "is_anomaly"]:
            if col in df.columns:
                labels = df[col]
                # If string labels, binarize
                if labels.dtype == object:
                    # Assume most frequent is normal
                    normal_val = labels.mode()[0]
                    labels = (labels != normal_val).astype(int)
                else:
                    labels = labels.astype(int)
                df_clean = df.drop(columns=[col])
                return df_clean, labels, col
        raise ValueError("Could not find a label column in the dataset")

# ──────────────────────────────────────────────────────────────────────
# Encoding & Scaling
# ──────────────────────────────────────────────────────────────────────

@timer
def encode_and_scale(df: pd.DataFrame):
    """
    Encode categorical columns with LabelEncoder, scale numerics with MinMaxScaler.
    Returns (scaled_array, scaler, label_encoders_dict, feature_names_list).
    """
    label_encoders = {}

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Drop IP address columns (not useful as features)
    ip_cols = [c for c in cat_cols if "ip" in c.lower() or "addr" in c.lower()]
    if ip_cols:
        df = df.drop(columns=ip_cols)
        cat_cols = [c for c in cat_cols if c not in ip_cols]
        logger.info(f"Dropped IP address columns: {ip_cols}")

    # Label encode remaining categoricals
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    if cat_cols:
        logger.info(f"Label-encoded {len(cat_cols)} categorical columns: {cat_cols}")

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(0, inplace=True)

    feature_names = df.columns.tolist()

    # Scale with MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.values)

    logger.info(f"Scaled features: {X_scaled.shape}")
    return X_scaled, scaler, label_encoders, feature_names

# ──────────────────────────────────────────────────────────────────────
# Train / Validation / Test Split
# ──────────────────────────────────────────────────────────────────────

@timer
def split_data(X: np.ndarray, y: np.ndarray, test_size=0.15, val_size=0.15, seed=42):
    """
    Split into train / validation / test sets (70/15/15 default).
    Stratifies on labels to preserve class distribution.
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Second split: train vs val
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=seed, stratify=y_temp
    )
    logger.info(
        f"Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ──────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────

@timer
def run_preprocessing_pipeline(filepath: str, test_size=0.15, val_size=0.15, seed=42):
    """
    End-to-end preprocessing pipeline:
      1. Load CSV
      2. Detect dataset format
      3. Clean data
      4. Extract labels
      5. Encode and scale features
      6. Split into train/val/test
      7. Save preprocessing artifacts
    Returns a dict with all splits and metadata.
    """
    # 1. Load
    if filepath.endswith(".pcap") or filepath.endswith(".pcapng"):
        df = load_pcap(filepath)
        dataset_format = "pcap"
    else:
        df = load_csv(filepath)
        dataset_format = detect_dataset_format(df)
    logger.info(f"Detected dataset format: {dataset_format}")

    # 2. Clean
    df = clean_data(df)

    # 3. Labels
    df_features, labels, label_col = extract_labels(df, dataset_format)

    # 4. Encode & Scale
    X_scaled, scaler, label_encoders, feature_names = encode_and_scale(df_features)
    y = labels.values

    # 5. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_scaled, y, test_size=test_size, val_size=val_size, seed=seed
    )

    # 6. Save artifacts
    save_preprocessing_artifacts(scaler, label_encoders, feature_names)

    result = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_names": feature_names,
        "dataset_format": dataset_format,
        "label_column": label_col,
        "n_features": X_scaled.shape[1],
        "class_distribution": {
            "normal": int((y == 0).sum()),
            "attack": int((y == 1).sum()),
        },
    }
    logger.info(f"Preprocessing complete. Classes: {result['class_distribution']}")
    return result


def generate_mock_dataset(format_type: str = "UNSW-NB15", n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a high-quality, realistic synthetic dataset for demo purposes.
    Supports 'UNSW-NB15' and 'CIC-IDS2017' formats.
    """
    np.random.seed(seed)
    
    if format_type == "UNSW-NB15":
        # Feature columns: sbytes, dbytes, sttl, dttl, dur, spkts, dpkts, service, proto, state, attack_cat, label
        # 85% normal, 15% attack
        n_attacks = int(n_samples * 0.15)
        n_normal = n_samples - n_attacks
        
        # Normal samples
        dur_norm = np.random.exponential(scale=0.5, size=n_normal) + 0.001
        sbytes_norm = np.random.lognormal(mean=6, sigma=1.5, size=n_normal).astype(int) + 40
        dbytes_norm = np.random.lognormal(mean=7, sigma=2.0, size=n_normal).astype(int) + 40
        sttl_norm = np.random.choice([31, 62, 254], size=n_normal, p=[0.1, 0.8, 0.1])
        dttl_norm = np.random.choice([29, 31, 252], size=n_normal, p=[0.05, 0.9, 0.05])
        spkts_norm = (sbytes_norm // 100) + np.random.randint(1, 10, size=n_normal)
        dpkts_norm = (dbytes_norm // 100) + np.random.randint(1, 10, size=n_normal)
        
        service_norm = np.random.choice(["http", "dns", "ssl", "ssh", "other"], size=n_normal, p=[0.4, 0.2, 0.2, 0.1, 0.1])
        proto_norm = np.random.choice(["tcp", "udp", "icmp"], size=n_normal, p=[0.7, 0.25, 0.05])
        state_norm = np.random.choice(["CON", "FIN", "INT", "REQ"], size=n_normal, p=[0.3, 0.5, 0.1, 0.1])
        
        # Attack samples
        dur_att = np.random.exponential(scale=1.5, size=n_attacks) + 0.001
        sbytes_att = np.random.lognormal(mean=12, sigma=2.5, size=n_attacks).astype(int) + 100
        dbytes_att = np.random.lognormal(mean=5, sigma=1.5, size=n_attacks).astype(int) + 20
        sttl_att = np.random.choice([62, 254], size=n_attacks, p=[0.2, 0.8])
        dttl_att = np.random.choice([0, 252], size=n_attacks, p=[0.7, 0.3])
        spkts_att = (sbytes_att // 1000) + np.random.randint(1, 50, size=n_attacks)
        dpkts_att = (dbytes_att // 1000) + np.random.randint(1, 20, size=n_attacks)
        
        service_att = np.random.choice(["http", "dns", "ssl", "ssh", "other"], size=n_attacks, p=[0.2, 0.1, 0.1, 0.1, 0.5])
        proto_att = np.random.choice(["tcp", "udp", "icmp"], size=n_attacks, p=[0.5, 0.4, 0.1])
        state_att = np.random.choice(["REQ", "INT", "RST"], size=n_attacks, p=[0.5, 0.4, 0.1])
        
        attack_cats = np.random.choice(["Exploits", "Fuzzers", "DoS", "Reconnaissance"], size=n_attacks, p=[0.4, 0.3, 0.2, 0.1])
        
        df_norm = pd.DataFrame({
            "dur": dur_norm, "sbytes": sbytes_norm, "dbytes": dbytes_norm,
            "sttl": sttl_norm, "dttl": dttl_norm, "spkts": spkts_norm, "dpkts": dpkts_norm,
            "service": service_norm, "proto": proto_norm, "state": state_norm,
            "attack_cat": "Normal", "label": 0
        })
        
        df_att = pd.DataFrame({
            "dur": dur_att, "sbytes": sbytes_att, "dbytes": dbytes_att,
            "sttl": sttl_att, "dttl": dttl_att, "spkts": spkts_att, "dpkts": dpkts_att,
            "service": service_att, "proto": proto_att, "state": state_att,
            "attack_cat": attack_cats, "label": 1
        })
        
        df = pd.concat([df_norm, df_att], ignore_index=True)
        # Shuffle
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return df
        
    elif format_type == "CIC-IDS2017":
        # Feature columns: Flow Duration, Total Fwd Packets, Total Backward Packets, Fwd Packet Length Max, Fwd Packet Length Min, Bwd Packet Length Max, Fwd Header Length, Label
        # 85% normal, 15% attack
        n_attacks = int(n_samples * 0.15)
        n_normal = n_samples - n_attacks
        
        # Normal
        dur_norm = np.random.lognormal(mean=10, sigma=3, size=n_normal).astype(int) + 10
        fwd_pkts_norm = np.random.randint(1, 20, size=n_normal)
        bwd_pkts_norm = np.random.randint(0, 30, size=n_normal)
        fwd_max_norm = np.random.lognormal(mean=5, sigma=1.0, size=n_normal).astype(int) + 20
        fwd_min_norm = np.random.randint(0, 64, size=n_normal)
        bwd_max_norm = np.random.lognormal(mean=6, sigma=1.5, size=n_normal).astype(int) + 20
        fwd_hdr_len_norm = fwd_pkts_norm * 20
        
        # Attack
        dur_att = np.random.lognormal(mean=4, sigma=2, size=n_attacks).astype(int) + 1
        fwd_pkts_att = np.random.randint(1, 100, size=n_attacks)
        bwd_pkts_att = np.random.randint(0, 10, size=n_attacks)
        fwd_max_att = np.random.lognormal(mean=8, sigma=2.0, size=n_attacks).astype(int) + 100
        fwd_min_att = np.random.randint(0, 20, size=n_attacks)
        bwd_max_att = np.random.lognormal(mean=3, sigma=1.0, size=n_attacks).astype(int) + 10
        fwd_hdr_len_att = fwd_pkts_att * 20
        
        labels_att = np.random.choice(["PORT_SCAN", "DOS_GOLDENEYE", "INFILTRATION"], size=n_attacks, p=[0.5, 0.3, 0.2])
        
        df_norm = pd.DataFrame({
            "Flow Duration": dur_norm, "Total Fwd Packets": fwd_pkts_norm,
            "Total Backward Packets": bwd_pkts_norm, "Fwd Packet Length Max": fwd_max_norm,
            "Fwd Packet Length Min": fwd_min_norm, "Bwd Packet Length Max": bwd_max_norm,
            "Fwd Header Length": fwd_hdr_len_norm, "Label": "BENIGN"
        })
        
        df_att = pd.DataFrame({
            "Flow Duration": dur_att, "Total Fwd Packets": fwd_pkts_att,
            "Total Backward Packets": bwd_pkts_att, "Fwd Packet Length Max": fwd_max_att,
            "Fwd Packet Length Min": fwd_min_att, "Bwd Packet Length Max": bwd_max_att,
            "Fwd Header Length": fwd_hdr_len_att, "Label": labels_att
        })
        
        df = pd.concat([df_norm, df_att], ignore_index=True)
        # Shuffle
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return df
    else:
        raise ValueError(f"Unknown mock format: {format_type}")

