"""
auth.py — Lightweight session-based authentication for the Streamlit app.
"""

import bcrypt
import yaml
import streamlit as st

CONFIG_PATH = "config.yaml"


def load_credentials():
    """Load user credentials from config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config.get("credentials", {})


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def login_form():
    """
    Render a login form. Sets st.session_state['authenticated'] on success.
    Returns True if user is authenticated, False otherwise.
    """
    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 80px auto;
        padding: 40px;
        background-color: #ffffff;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 20px 45px rgba(99, 102, 241, 0.08);
    }
    .login-title {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: #4f46e5;
        margin-bottom: 8px;
    }
    .login-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<p class="login-title">🛡️ NetShield</p>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">ML Network Anomaly Detection System</p>', unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                credentials = load_credentials()
                if username in credentials:
                    stored_hash = credentials[username]["password"]
                    if verify_password(password, stored_hash):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.session_state["display_name"] = credentials[username].get("name", username)
                        st.rerun()
                    else:
                        st.error("Invalid password")
                else:
                    st.error("User not found")

        st.markdown("---")
        st.caption("Default credentials: `admin` / `admin`")

    return False


def logout():
    """Clear session state and log out."""
    for key in ["authenticated", "username", "display_name"]:
        st.session_state.pop(key, None)
    st.rerun()


def require_auth(func):
    """Decorator: show login if not authenticated, else run the page."""
    def wrapper(*args, **kwargs):
        if not login_form():
            st.stop()
        return func(*args, **kwargs)
    return wrapper
