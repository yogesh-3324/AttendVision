"""
pages/capture.py  — Take Attendance
Upload classroom photo → detect faces → match → show results → send emails
"""

import streamlit as st
import cv2
import numpy as np
import uuid
import os
from datetime import datetime
from PIL import Image

import database as db
import face_engine as fe
import email_service as em
from config import SCHOOL_NAME, UPLOADS_DIR


def show():
    st.title("📷 Take Attendance")
    st.markdown("Upload a classroom photo. The system will detect every face and match them against enrolled students.")

    # ── Setup ──────────────────────────────────────────────────────────────────
    classes = db.get_all_classes()
    if not classes:
        st.warning("No students enrolled yet. Go to **Enroll Students** first.")
        return

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        class_name = st.selectbox("Class", classes)
    with col2:
        subject = st.text_input("Subject", placeholder="e.g. Data Structures")
    with col3:
        teacher_name = st.text_input("Teacher", placeholder="Optional")

    uploaded = st.file_uploader(
        "Upload classroom photo",
        type=["jpg", "jpeg", "png"],
        help="Best results: good lighting, faces visible, photo taken from front of class",
    )

    if uploaded is None:
        _show_tips()
        return

    # ── Run pipeline ───────────────────────────────────────────────────────────
    img_bytes = uploaded.read()
    img_bgr   = fe.bytes_to_bgr(img_bytes)

    if img_bgr is None:
        st.error("Could not read image. Please upload a valid JPEG or PNG.")
        return

    students = db.get_all_students(class_name)
    if not students:
        st.error(f"No students found in class **{class_name}**.")
        return

    enrolled = [s for s in students if os.path.exists(
        os.path.join(__import__('config').EMBEDDINGS_DIR, f"{s['id']}.npy")
    )]

    st.info(f"**{class_name}** · {len(enrolled)} enrolled students · Processing image…")

    # Live progress bar
    progress = st.progress(0, text="Step 1/4 — Detecting faces…")

    result = None
    with st.spinner(""):
        # Step 1 — detect
        progress.progress(15, text="Step 1/4 — Detecting faces with RetinaFace…")
        faces = fe.detect_faces(img_bgr)
        progress.progress(35, text=f"Step 2/4 — Found {len(faces)} face(s). Generating ArcFace embeddings…")

        # Step 2+3 — embed + match
        db_embeddings = fe.load_all_embeddings(enrolled)
        present_ids = set()
        matches = []

        if faces and db_embeddings:
            embeddings = []
            valid_faces = []
            for face in faces:
                try:
                    embeddings.append(fe.get_embedding(face.aligned))
                    valid_faces.append(face)
                except Exception:
                    pass

            progress.progress(60, text="Step 3/4 — Running 1:N cosine matching…")
            matches = fe.identify_batch(embeddings, db_embeddings)

            for m in matches:
                if m.is_known and m.student_id:
                    present_ids.add(m.student_id)

        progress.progress(80, text="Step 4/4 — Building annotated image…")
        annotated = fe._draw_results(img_bgr.copy(), faces, matches)

        # Save classroom image
        img_filename = f"classroom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        img_path = os.path.join(UPLOADS_DIR, img_filename)
        cv2.imwrite(img_path, img_bgr)

        progress.progress(100, text="Done!")

    progress.empty()

    # ── Show annotated image ───────────────────────────────────────────────────
    st.image(
        fe.bgr_to_rgb(annotated),
        caption=f"Detected {len(faces)} face(s) · {len(present_ids)} recognized · {sum(1 for m in matches if not m.is_known)} unknown",
        use_column_width=True,
    )

    # ── Build full attendance table ────────────────────────────────────────────
    present_map = {}
    conf_map    = {}
    for face, match in zip(faces, matches):
        if match.is_known and match.student_id:
            present_map[match.student_id] = match
            conf_map[match.student_id]    = match.confidence

    all_records = []
    for s in students:
        if s["id"] in present_map:
            m = present_map[s["id"]]
            all_records.append({
                "name": s["name"], "roll": s["roll_number"], "email": s["email"],
                "status": "✅ Present",
                "status_raw": "present",
                "confidence": f"{m.confidence*100:.1f}%",
                "student_id": s["id"],
            })
        else:
            all_records.append({
                "name": s["name"], "roll": s["roll_number"], "email": s["email"],
                "status": "❌ Absent",
                "status_raw": "absent",
                "confidence": "—",
                "student_id": s["id"],
            })

    # Summary metrics
    n_present = sum(1 for r in all_records if r["status_raw"] == "present")
    n_absent  = len(all_records) - n_present
    n_unknown = sum(1 for m in matches if not m.is_known)
    pct       = round(n_present / len(all_records) * 100) if all_records else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students", len(all_records))
    m2.metric("Present",        n_present,  delta=f"{pct}%")
    m3.metric("Absent",         n_absent)
    m4.metric("Unknown Faces",  n_unknown)

    st.markdown("#### Attendance Sheet")
    import pandas as pd
    df = pd.DataFrame([
        {"Roll No.": r["roll"], "Name": r["name"], "Status": r["status"], "Confidence": r["confidence"]}
        for r in all_records
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Unknown face thumbnails ────────────────────────────────────────────────
    unknown_faces = [f for f, m in zip(faces, matches) if not m.is_known]
    if unknown_faces:
        st.markdown(f"#### {len(unknown_faces)} Unrecognized Face(s)")
        st.caption("These faces were detected but didn't match any enrolled student (cosine distance > threshold).")
        cols = st.columns(min(len(unknown_faces), 6))
        for i, face in enumerate(unknown_faces[:6]):
            with cols[i]:
                st.image(fe.bgr_to_rgb(face.crop), caption=f"Unknown #{i+1}", use_column_width=True)

    # ── Save session + send emails ─────────────────────────────────────────────
    session_id  = str(uuid.uuid4())
    session_date = datetime.now().strftime("%b %d, %Y · %I:%M %p")

    col_save, col_email = st.columns(2)

    with col_save:
        if st.button("💾 Save Attendance", type="primary", use_container_width=True):
            db.save_session(
                session_id    = session_id,
                class_name    = class_name,
                session_date  = session_date,
                subject       = subject or "Class",
                teacher_name  = teacher_name,
                image_path    = img_path,
                total_detected   = len(faces),
                total_recognized = n_present,
                total_absent     = n_absent,
            )
            for rec in all_records:
                match = present_map.get(rec["student_id"])
                db.save_record(
                    record_id  = str(uuid.uuid4()),
                    session_id = session_id,
                    student_id = rec["student_id"],
                    status     = rec["status_raw"],
                    confidence = match.confidence if match else None,
                )
            st.success("Attendance saved successfully!")
            st.session_state["last_session_id"] = session_id

    with col_email:
        if st.button("📧 Save & Send Emails", use_container_width=True):
            # Save first
            db.save_session(
                session_id = session_id,
                class_name = class_name,
                session_date = session_date,
                subject = subject or "Class",
                teacher_name = teacher_name,
                image_path = img_path,
                total_detected   = len(faces),
                total_recognized = n_present,
                total_absent     = n_absent,
            )
            for rec in all_records:
                match = present_map.get(rec["student_id"])
                db.save_record(
                    record_id  = str(uuid.uuid4()),
                    session_id = session_id,
                    student_id = rec["student_id"],
                    status     = rec["status_raw"],
                    confidence = match.confidence if match else None,
                )

            # Build email payloads with per-student attendance %
            email_records = []
            for rec in all_records:
                att_pct = db.get_student_attendance_percent(rec["student_id"], class_name)
                email_records.append({
                    "name":               rec["name"],
                    "email":              rec["email"],
                    "roll_number":        rec["roll"],
                    "status":             rec["status_raw"],
                    "confidence":         conf_map.get(rec["student_id"]),
                    "attendance_percent": att_pct,
                })

            with st.spinner(f"Sending {len(email_records)} emails…"):
                sent, failed = em.send_attendance_emails(
                    email_records,
                    {"class_name": class_name, "session_date": session_date, "subject": subject or "Class"},
                )
            db.mark_emails_sent(session_id)
            st.success(f"Emails sent: {sent} ✓  Failed: {failed}")
            st.session_state["last_session_id"] = session_id


def _show_tips():
    st.markdown("---")
    st.markdown("#### Tips for best accuracy")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📸 Photo capture**\n- Take from front of class\n- Ensure faces are visible\n- Good lighting required\n- Avoid backlighting")
    with c2:
        st.markdown("**👤 Enrollment**\n- 3–5 photos per student\n- Vary angles slightly\n- Include glasses if worn\n- Use clear, well-lit shots")
    with c3:
        st.markdown("**⚙️ Settings**\n- Threshold 0.4 = balanced\n- Lower = stricter matching\n- Check `.env` for config\n- Buffalo_l = best accuracy")
