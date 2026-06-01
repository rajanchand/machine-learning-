# 🛡️ NetShield Enterprise — Advanced NIDS & Threat Intelligence Dashboard

A production-grade, dissertation-quality **Network Intrusion Detection System (NIDS)** with a secure Streamlit web interface. Engineered to 2026 standards, this platform integrates robust deep learning models, strict cryptographic security bounds, live network packet sniffing, and comprehensive evaluation capabilities.

---

## 🌟 Advanced Features

### 1. Multi-Dataset Support (2026 Benchmarks)
Configurable and autodetected support for 5 major NIDS benchmarks:
- **UNSW-NB15** — Flow-based network profiles
- **CIC-IDS2017** — Standard benchmark features
- **CSE-CIC-IDS2018** — Large-scale enterprise flows
- **Edge-IIoTset** — Modern IoT/IIoT attack surfaces
- **WUSTL-IIoT-2021** — Industrial control systems protocols

### 2. High-Performance Models & Methodology
- **Baselines**: XGBoost (Supervised), Isolation Forest (Unsupervised), One-Class SVM (Unsupervised RBF)
- **Proposed Architectures**:
  - **LSTM Autoencoder** with optional Soft Attention mechanism.
  - **Proposed Hybrid 1D-CNN-LSTM Autoencoder with Soft Attention**: Uses 1D spatial convolutions to capture structural associations across tabular fields, temporal LSTM sequence layers, and additive soft sequential attention.
- **Adaptive Score Fusion**: Min-Max score normalization combined via weighted average and bounded by an **Adaptive Fusion Threshold** (Gaussian std-dev + extreme percentile estimators).

### 3. Strict Enterprise Security Boundaries (OWASP Compliant)
- **Model Signature Auditing (SHA-256)**: Every model checkpoint is digitally signed with a sidecar hash file. Any model load triggers verification; tampered models immediately throw a `ModelSignatureException` and block inference.
- **Secure Sandbox Uploads**: Sandboxed file processing limits memory, parses MIME magic bytes to block file-type spoofing, runs malicious hash antivirus simulations, and sanitizes paths.
- **JWT Simulated Auth**: Lightweight bcrypt credentials (`config.yaml`) with simulated stateful JWT tokens and a **15-minute inactivity session auto-timeout**.
- **OWASP Security Audit Log**: All security-relevant events (logins, uploads, verification failures) write to structured audit records.

### 4. Real-Time Scapy Sniffing & Flow Analysis
- **Live Packet Capture**: Multi-threaded packet capture utilizing background Scapy sniffers writing to thread-safe queues.
- **High-Fidelity Simulation Fallback**: Gracefully detects privilege restrictions (e.g. non-root Streamlit deployment) and boots a lifelike simulated live flow stream (port sweeps, DDoS, benign flows) to ensure the interface remains fully operational and interactive anywhere.
- **System Metrics**: Real-time packet flow rates (packets/sec) and CPU/RAM gauges.

### 5. Cohesive Light-Theme Visualizations & Reporting
- Styled with a premium, crisp Light Theme (Indigo/Slate palette).
- Includes radar model comparisons, dynamic ablation studies, and comprehensive HTML report generation featuring a digital dissertation seal and cryogenic signatures.

---

## 🚀 Quick Start

### Local Virtual Environment

```bash
# 1. Clone & enter repository
git clone https://github.com/rajanchand/machine-learning-.git
cd machine-learning-

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Start dashboard
streamlit run app.py
```

Default credentials: `admin` / `admin`

---

## 🐳 Docker Deployment (Production Sandboxed)

Deploy in a secure, non-root, resource-capped container environment utilizing Docker Compose:

```bash
# 1. Create a copy of environment variables
cp .env.example .env

# 2. Spin up the dashboard container
docker-compose up --build -d

# 3. Check health status
docker-compose ps
```

The container applies a strict **1.5 CPU Cap** and **1.5 GB Memory Limit** to mitigate DDoS and resources exhaustion vectors, running securely as the unprivileged system user `nids`.

---

## 📁 Project Structure

```
├── app.py                      # Main entrypoint & session-timer injection
├── config.yaml                 # Cryptographic user password & auth bounds
├── requirements.txt            # Assembled dependencies
├── Dockerfile                  # Secure Multi-Stage Docker Build
├── docker-compose.yml          # Container configuration & CPU/RAM limits
├── .env.example                # Config template variables
├── models/saved/               # Persistent directory for signed models
├── streamlit_app/              # UI Presentation Layer
│   ├── auth.py                 # JWT simulated auth & timeout bounds
│   ├── security.py             # Sandbox processing, antivirus, rate-limiter
│   ├── pages/                  # Streamlit Page Controllers
│   │   ├── 1_dashboard.py
│   │   ├── 2_data_explorer.py  # Sandboxed file parsing & synthetic demos
│   │   ├── 3_training.py       # Advanced preprocessors, OCSVM, CNN-LSTM-AE
│   │   ├── 4_prediction.py     # Scapy live sniffing & digital seals verification
│   │   ├── 5_results.py        # Confusion matrices & interactive Plotly ROC
│   │   ├── 6_comparison.py     # Side-by-side radars & dynamic ablation studies
│   │   ├── 7_explainability.py # SHAP interpretability graphs
│   │   └── 8_report.py         # Light-theme PDF/HTML export and integrity seals
│   └── components/             # Reusable Styles and Plotting elements
├── src/                        # Core Machine Learning Pipeline
│   ├── preprocessing.py        # Scalers, MI selectors, PCA, datasets
│   ├── evaluation.py           # Metrics and classification reports
│   ├── explainability.py       # SHAP importance calculations
│   ├── report_generator.py     # HTML report compiler
│   ├── utils.py                # Model signatures, secure saving/loading, logging
│   └── models/                 # Model files
```

---

## 🔒 Verification & Security Controls Audit

All operations are securely monitored and recorded. You can view structured audit records in your execution terminal or system logs:

* **[AUDIT] Event: MODEL_LOAD | Status: SUCCESS** — Cryptographic signature checked and matched SHA-256 sidecar.
* **[AUDIT] Event: UNTRUSTED_UPLOAD | Status: BLOCKED** — Invalid magic bytes or file signature spoof detected.
* **[AUDIT] Event: SESSION_TIMEOUT | Status: EXPIRED** — Active token invalidated due to 15-minute inactivity boundary.
