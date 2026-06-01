"""
styles.py — Custom CSS for a professional, minimalist dark-themed UI.
"""

import streamlit as st


def inject_custom_css():
    """Inject global custom CSS for consistent theming."""
    st.markdown("""
    <style>
    /* ── Global ────────────────────────────────── */
    @import url('https://fonts.g    html, body, div.stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #ffffff;
        color: #0f172a;
    }

    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* ── Metric Cards ─────────────────────────── */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.12), 0 4px 6px -4px rgba(99, 102, 241, 0.12);
        border-color: #cbd5e1;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #4f46e5;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-delta-pos { color: #16a34a; font-size: 14px; font-weight: 600; }
    .metric-delta-neg { color: #dc2626; font-size: 14px; font-weight: 600; }

    /* ── Section Headers ──────────────────────── */
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #4f46e5;
        margin: 30px 0 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #6366f1;
    }

    /* ── Status Badges ────────────────────────── */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-success { background: #dcfce7; color: #16a34a; }
    .badge-warning { background: #fef9c3; color: #ca8a04; }
    .badge-danger  { background: #fee2e2; color: #dc2626; }
    .badge-info    { background: #dbeafe; color: #2563eb; }

    /* ── Tables ────────────────────────────────── */
    .stDataFrame, .stTable {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* ── Sidebar ──────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: #4f46e5;
        font-size: 22px;
    }

    /* Beautiful Human-made Menu for Sidebar Radio Buttons */
    [data-testid="stSidebar"] [role="radiogroup"] {
        background-color: transparent;
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding: 10px 0;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background-color: transparent !important;
        border-radius: 8px;
        padding: 10px 14px !important;
        margin: 0 !important;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
        display: flex;
        align-items: center;
        width: 100%;
        border: 1px solid transparent;
    }
    /* Hide default radio circle completely */
    [data-testid="stSidebar"] [role="radiogroup"] label div[role="presentation"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label span:first-child {
        display: none !important;
    }
    /* Style option labels */
    [data-testid="stSidebar"] [role="radiogroup"] label p {
        color: #475569 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: rgba(99, 102, 241, 0.05) !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background-color: rgba(99, 102, 241, 0.08) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] p {
        color: #4f46e5 !important;
        font-weight: 600 !important;
    }

    /* ── Buttons ──────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 24px;
        transition: all 0.2s;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
        color: white;
    }

    /* ── Tabs ─────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }

    /* ── Expanders ────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #4f46e5;
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value, delta: str = None):
    """Render a styled metric card as HTML."""
    delta_html = ""
    if delta:
        css = "metric-delta-pos" if not delta.startswith("-") else "metric-delta-neg"
        delta_html = f'<div class="{css}">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def section_header(text: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def badge(text: str, variant: str = "info"):
    """Render a status badge. variant: success, warning, danger, info."""
    return f'<span class="badge badge-{variant}">{text}</span>'
