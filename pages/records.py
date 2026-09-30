"""
pages/records.py  — Attendance Records
Browse past sessions, view per-student records, resend emails.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

import database as db
import email_service as em


def show():
    st.title("📋 Attendance Records")

    classes = db.get_all_classes()
    if not classes:
        st.info("No attendance sessions yet.")
        return

    col1, col2 = st.columns([2, 4])
    with col1:
        filter_class = st.selectbox("Class", ["All"] + classes, key="records_class")

    sessions = db.get_recent_sessions(
        class_name=filter_class if filter_class != "All" else None,
        limit=50,
    )

    if not sessions:
        st.info("No sessions found.")
        return

    # Sessions list
    st.markdown(f"**{len(sessions)} session(s) found**")

    sessions_df = pd.DataFrame([
        {
            "Date":       s["session_date"],
            "Class":      s["class_name"],
            "Subject":    s.get("subject") or "—",
            "Detected":   s["total_detected"],
            "Present":    s["total_recognized"],
            "Absent":     s["total_absent"],
            "Rate":       f"{round(s['total_recognized'] / max(s['total_recognized'] + s['total_absent'], 1) * 100)}%",
            "Emails":     "✅ Sent" if s["emails_sent"] else "⏳ Pending",
            "ID":         s["id"],
        }
        for s in sessions
    ])
    st.dataframe(
        sessions_df.drop(columns=["ID"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Session detail ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### View Session Detail")

    session_options = {
        f"{s['session_date']}  ·  {s['class_name']}  ·  {s.get('subject','Class')}": s["id"]
        for s in sessions
    }
    selected_label = st.selectbox("Select session", list(session_options.keys()), key="session_select")
    selected_id    = session_options[selected_label]

    session  = db.get_session(selected_id)
    records  = db.get_session_records(selected_id)

    if not records:
        st.info("No records for this session.")
        return

    # Metrics
    present = sum(1 for r in records if r["status"] == "present")
    absent  = len(records) - present
    pct     = round(present / len(records) * 100) if records else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total",   len(records))
    m2.metric("Present", present)
    m3.metric("Absent",  absent)
    m4.metric("Rate",    f"{pct}%")

    # Records table
    df = pd.DataFrame([
        {
            "Roll No.":  r.get("roll_number") or "—",
            "Name":      r.get("name") or "Unknown",
            "Status":    "✅ Present" if r["status"] == "present" else "❌ Absent",
            "Confidence": f"{round(r['confidence']*100, 1)}%" if r.get("confidence") else "—",
            "Email":     r.get("email") or "—",
            "Email Sent": "✅" if r.get("email_sent") else "—",
        }
        for r in records
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        csv,
        file_name=f"attendance_{session['class_name']}_{session['session_date'][:10]}.csv",
        mime="text/csv",
    )

    # ── Resend emails ──────────────────────────────────────────────────────────
    st.markdown("---")
    if session.get("emails_sent"):
        st.success("Emails were already sent for this session.")
        if st.button("📧 Resend Emails (override)", key="resend"):
            _do_send_emails(selected_id, records, session)
    else:
        if st.button("📧 Send Emails Now", type="primary", key="send_emails"):
            _do_send_emails(selected_id, records, session)


def _do_send_emails(session_id, records, session):
    email_records = []
    for r in records:
        if not r.get("email"):
            continue
        att_pct = db.get_student_attendance_percent(r.get("student_id"), session["class_name"])
        email_records.append({
            "name":               r.get("name", "Student"),
            "email":              r["email"],
            "roll_number":        r.get("roll_number", ""),
            "status":             r["status"],
            "confidence":         r.get("confidence"),
            "attendance_percent": att_pct,
        })

    with st.spinner(f"Sending {len(email_records)} emails…"):
        sent, failed = em.send_attendance_emails(
            email_records,
            {
                "class_name":   session["class_name"],
                "session_date": session["session_date"],
                "subject":      session.get("subject") or "Class",
            },
        )

    db.mark_emails_sent(session_id)
    if failed == 0:
        st.success(f"All {sent} emails sent successfully!")
    else:
        st.warning(f"Sent: {sent}  ·  Failed: {failed}  (check SMTP config in .env)")
