"""
pages/reports.py  — Analytics & Reports
Weekly trends, per-student history, low-attendance alerts.
"""

import streamlit as st
import pandas as pd

import database as db


def show():
    st.title("📊 Reports & Analytics")

    classes = db.get_all_classes()
    if not classes:
        st.info("No data yet. Enroll students and take attendance first.")
        return

    class_name = st.selectbox("Select class", classes, key="reports_class")
    sessions   = db.get_recent_sessions(class_name=class_name, limit=100)

    if not sessions:
        st.info(f"No attendance sessions recorded for **{class_name}** yet.")
        return

    # ── Overview metrics ───────────────────────────────────────────────────────
    total_sessions = len(sessions)
    total_present  = sum(s["total_recognized"] for s in sessions)
    total_records  = sum(s["total_recognized"] + s["total_absent"] for s in sessions)
    avg_rate       = round(total_present / total_records * 100, 1) if total_records > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sessions", total_sessions)
    m2.metric("Avg Attendance", f"{avg_rate}%")
    m3.metric("Total Present",  total_present)
    m4.metric("Total Absent",   total_records - total_present)

    # ── Attendance trend chart ─────────────────────────────────────────────────
    st.markdown("#### Attendance Trend")
    chart_data = pd.DataFrame([
        {
            "Date": s["session_date"][:10],
            "Present": s["total_recognized"],
            "Absent":  s["total_absent"],
            "Rate %":  round(s["total_recognized"] / max(s["total_recognized"] + s["total_absent"], 1) * 100, 1),
        }
        for s in reversed(sessions[:30])   # oldest → newest
    ])
    chart_data["Date"] = pd.to_datetime(chart_data["Date"], format="mixed", errors="coerce")
    chart_data = chart_data.sort_values("Date")

    tab_bar, tab_line = st.tabs(["Present / Absent", "Attendance Rate %"])
    with tab_bar:
        st.bar_chart(chart_data.set_index("Date")[["Present", "Absent"]], use_container_width=True)
    with tab_line:
        st.line_chart(chart_data.set_index("Date")[["Rate %"]], use_container_width=True)

    # ── Per-student summary ────────────────────────────────────────────────────
    st.markdown("#### Per-Student Attendance Summary")
    students = db.get_all_students(class_name)

    rows = []
    for s in students:
        pct = db.get_student_attendance_percent(s["id"], class_name)
        history = db.get_student_attendance_history(s["id"], limit=100)
        present_count = sum(1 for h in history if h["status"] == "present")
        total_count   = len(history)
        rows.append({
            "Roll No.":        s["roll_number"],
            "Name":            s["name"],
            "Sessions":        total_count,
            "Present":         present_count,
            "Absent":          total_count - present_count,
            "Attendance %":    f"{pct}%" if pct is not None else "—",
            "_pct":            pct or 0,
            "_id":             s["id"],
        })

    rows.sort(key=lambda x: x["_pct"])

    df_students = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in rows
    ])
    st.dataframe(df_students, use_container_width=True, hide_index=True)

    # ── Low attendance alert ───────────────────────────────────────────────────
    threshold = st.slider("Low attendance threshold (%)", 50, 90, 75, step=5)
    low_students = [r for r in rows if r["_pct"] < threshold and r["Sessions"] > 0]

    if low_students:
        st.error(f"⚠️ {len(low_students)} student(s) below {threshold}% attendance")
        df_low = pd.DataFrame([
            {
                "Roll No.":     r["Roll No."],
                "Name":         r["Name"],
                "Attendance %": r["Attendance %"],
                "Sessions":     r["Sessions"],
            }
            for r in low_students
        ])
        st.dataframe(df_low, use_container_width=True, hide_index=True)

        csv_low = df_low.to_csv(index=False)
        st.download_button(
            "⬇️ Download Low Attendance List",
            csv_low,
            file_name=f"low_attendance_{class_name}.csv",
            mime="text/csv",
        )
    else:
        st.success(f"All students are above {threshold}% attendance.")

    # ── Individual student drill-down ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Individual Student History")

    student_options = {f"{s['roll_number']} — {s['name']}": s["id"] for s in students}
    if student_options:
        selected = st.selectbox("Select student", list(student_options.keys()), key="student_drill")
        student_id = student_options[selected]

        history = db.get_student_attendance_history(student_id, limit=50)
        if history:
            df_hist = pd.DataFrame([
                {
                    "Date":       h["session_date"],
                    "Subject":    h.get("subject") or "—",
                    "Status":     "✅ Present" if h["status"] == "present" else "❌ Absent",
                    "Confidence": f"{round(h['confidence']*100, 1)}%" if h.get("confidence") else "—",
                }
                for h in history
            ])
            pct = db.get_student_attendance_percent(student_id, class_name)
            st.caption(f"Overall attendance in {class_name}: **{pct}%**" if pct else "No data")
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("No attendance history for this student.")
