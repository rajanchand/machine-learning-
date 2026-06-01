"""
auth.py — Simulated OAuth/JWT-style authentication and session timeout controls.
"""

import bcrypt
import yaml
import time
import base64
import streamlit as st
from src.utils import security_audit_log

CONFIG_PATH = "config.yaml"
SESSION_TIMEOUT_SECONDS = 900  # 15 minutes of inactivity


def load_credentials():
    """Load user credentials from config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config.get("credentials", {})


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def generate_mock_jwt(username: str) -> str:
    """Generate a simulated base64 JWT token containing session details."""
    payload = f"iss:netshield|sub:{username}|iat:{int(time.time())}"
    token_bytes = base64.b64encode(payload.encode())
    return f"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.{token_bytes.decode()}.secure_sig"


def check_session_timeout() -> bool:
    """
    Check if the authenticated session has expired due to inactivity.
    Returns True if session is valid, False if timed out.
    """
    if not st.session_state.get("authenticated"):
        return True
        
    last_act = st.session_state.get("last_activity", 0.0)
    current_time = time.time()
    
    if current_time - last_act > SESSION_TIMEOUT_SECONDS:
        # Audit Log
        security_audit_log(
            "SESSION_TIMEOUT",
            st.session_state.get("username", "unknown"),
            "SUCCESS",
            "User session automatically timed out due to 15 minutes of inactivity",
            "INFO"
        )
        logout(timeout=True)
        return False
        
    # Update last activity timestamp
    st.session_state["last_activity"] = current_time
    return True


def login_form():
    """
    Render a secure login form with simulated JWT token registration.
    """
    # Verify timeout first
    if st.session_state.get("authenticated"):
        if not check_session_timeout():
            return False
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

        if st.session_state.pop("session_timeout_msg", False):
            st.warning("⏱️ Session expired due to inactivity. Please sign in again.")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                # Sanitise inputs
                username = username.strip()[:50]
                
                credentials = load_credentials()
                if username in credentials:
                    stored_hash = credentials[username]["password"]
                    if verify_password(password, stored_hash):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.session_state["display_name"] = credentials[username].get("name", username)
                        st.session_state["token"] = generate_mock_jwt(username)
                        st.session_state["last_activity"] = time.time()
                        
                        security_audit_log(
                            "USER_LOGIN",
                            username,
                            "SUCCESS",
                            "User logged in successfully and JWT token was issued."
                        )
                        st.rerun()
                    else:
                        security_audit_log(
                            "USER_LOGIN",
                            username,
                            "FAILURE",
                            "Failed login attempt: Invalid password.",
                            "WARNING"
                        )
                        st.error("Invalid password")
                else:
                    security_audit_log(
                        "USER_LOGIN",
                        username,
                        "FAILURE",
                        "Failed login attempt: User not found.",
                        "WARNING"
                    )
                    st.error("User not found")

        st.markdown("---")
        st.caption("Default credentials: `admin` / `admin`")

    return False


def logout(timeout: bool = False):
    """Clear session state and log out."""
    username = st.session_state.get("username", "unknown")
    if not timeout:
        security_audit_log(
            "USER_LOGOUT",
            username,
            "SUCCESS",
            "User logged out manually."
        )
        
    for key in ["authenticated", "username", "display_name", "token", "last_activity"]:
        st.session_state.pop(key, None)
        
    if timeout:
        st.session_state["session_timeout_msg"] = True
        
    st.rerun()


def require_auth(func):
    """Decorator: show login if not authenticated, else run the page."""
    def wrapper(*args, **kwargs):
        if not login_form():
            st.stop()
        return func(*args, **kwargs)
    return wrapper
