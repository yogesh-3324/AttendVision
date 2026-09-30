"""
app.py  — SmartAttend main Streamlit app
Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="SmartAttend",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p { color: #94a3b8 !important; }
[data-testid="stSidebarNav"] { display: none; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
}
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

import database  # ensures DB is initialised

# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 SmartAttend")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📷  Take Attendance", "👤  Enroll Students", "📋  Attendance Records", "📊  Reports"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>Face Recognition<br>Attendance System<br>v1.0</small>",
        unsafe_allow_html=True,
    )

import importlib

# ── Route to page ─────────────────────────────────────────────────────────────
if "📷" in page:
    from pages import capture
    importlib.reload(capture)
    capture.show()
elif "👤" in page:
    from pages import enroll
    importlib.reload(enroll)
    enroll.show()
elif "📋" in page:
    from pages import records
    importlib.reload(records)
    records.show()
elif "📊" in page:
    from pages import reports
    importlib.reload(reports)
    reports.show()
