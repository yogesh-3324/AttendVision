"""
pages/enroll.py  — Student Enrollment
Add new students and compute their ArcFace embeddings from uploaded photos.
"""

import streamlit as st
import os
import uuid
import cv2
from PIL import Image
import numpy as np

import database as db
import face_engine as fe
from config import EMBEDDINGS_DIR, UPLOADS_DIR


def show():
    st.title("👤 Enroll Students")

    tab1, tab2 = st.tabs(["➕ Add New Student", "📋 Manage Enrolled Students"])

    with tab1:
        _enroll_form()

    with tab2:
        _manage_students()


def _enroll_form():
    st.markdown("Upload **3–5 clear face photos** per student for best recognition accuracy.")

    with st.form("enroll_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            name        = st.text_input("Full Name *", placeholder="Arjun Sharma")
            roll_number = st.text_input("Roll Number *", placeholder="CS2301")
            class_name  = st.text_input("Class *", placeholder="CS-301")
        with c2:
            email   = st.text_input("Email *", placeholder="arjun@college.edu")
            section = st.text_input("Section", placeholder="A")
            phone   = st.text_input("Phone", placeholder="9876543210")

        photos = st.file_uploader(
            "Face Photos * (3–5 recommended)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Use clear, well-lit photos. Each photo should contain exactly one face.",
        )

        submitted = st.form_submit_button("Enroll Student", type="primary", use_container_width=True)

    if not submitted:
        return

    # Validate
    errors = []
    if not name.strip():        errors.append("Name is required")
    if not roll_number.strip(): errors.append("Roll Number is required")
    if not email.strip():       errors.append("Email is required")
    if not class_name.strip():  errors.append("Class is required")
    if not photos:              errors.append("At least 1 photo is required")

    if db.get_student_by_roll(roll_number.strip()):
        errors.append(f"Roll number **{roll_number.strip()}** is already enrolled")

    existing_by_email = db.get_student_by_email(email.strip())
    if existing_by_email:
        errors.append(f"Email **{email.strip()}** is already enrolled for student **{existing_by_email['name']}** (Roll No: **{existing_by_email['roll_number']}**)")

    if errors:
        for e in errors:
            st.error(e)
        return

    # Process photos
    student_id = str(uuid.uuid4())
    aligned_faces = []
    failed_photos = []
    preview_crops = []

    progress = st.progress(0, text="Processing photos…")

    for i, photo_file in enumerate(photos[:10]):
        progress.progress(int((i + 1) / len(photos) * 70), text=f"Processing photo {i+1}/{len(photos)}…")
        img_bytes = photo_file.read()
        img_bgr   = fe.bytes_to_bgr(img_bytes)

        if img_bgr is None:
            failed_photos.append(f"Photo {i+1}: could not read file")
            continue

        faces = fe.detect_faces(img_bgr)
        if not faces:
            failed_photos.append(f"Photo {i+1}: no face detected")
            continue
        if len(faces) > 1:
            # Use the largest face
            faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
            st.warning(f"Photo {i+1}: multiple faces found — using the largest one")

        aligned_faces.append(faces[0].aligned)
        preview_crops.append(fe.bgr_to_rgb(faces[0].crop))

    if not aligned_faces:
        st.error("Could not extract any faces. Please upload clear, well-lit photos.")
        for msg in failed_photos:
            st.warning(msg)
        return

    progress.progress(85, text="Computing ArcFace embedding…")

    try:
        fe.enroll_student(student_id, aligned_faces)
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return

    progress.progress(100, text="Saving to database…")

    # Save profile photo (first successful crop)
    profile_path = None
    if preview_crops:
        profile_path = os.path.join(UPLOADS_DIR, f"profile_{student_id}.jpg")
        import cv2 as cv
        cv.imwrite(profile_path, fe.bytes_to_bgr(photos[0].getvalue()) if aligned_faces else None)

    try:
        db.add_student(
            student_id   = student_id,
            roll_number  = roll_number.strip(),
            name         = name.strip(),
            email        = email.strip().lower(),
            class_name   = class_name.strip(),
            section      = section.strip() or None,
            phone        = phone.strip() or None,
            photo_path   = profile_path,
            photos_count = len(aligned_faces),
        )
    except Exception as e:
        progress.empty()
        st.error(f"Failed to save student record: {e}")
        return

    progress.empty()
    st.success(f"✅ **{name}** enrolled successfully using {len(aligned_faces)} photo(s)!")

    if failed_photos:
        st.warning(f"{len(failed_photos)} photo(s) skipped:")
        for msg in failed_photos:
            st.caption(f"  • {msg}")

    # Show face crops used
    if preview_crops:
        st.markdown("**Faces extracted and used for enrollment:**")
        cols = st.columns(min(len(preview_crops), 5))
        for i, crop in enumerate(preview_crops[:5]):
            with cols[i]:
                st.image(crop, caption=f"Photo {i+1}", use_column_width=True)


def _manage_students():
    classes = db.get_all_classes()

    filter_class = st.selectbox(
        "Filter by class", ["All"] + classes, key="manage_class_filter"
    )

    students = db.get_all_students(filter_class if filter_class != "All" else None)

    if not students:
        st.info("No students enrolled yet.")
        return

    st.caption(f"{len(students)} student(s) enrolled")

    # Table
    import pandas as pd
    rows = []
    for s in students:
        has_emb = os.path.exists(os.path.join(EMBEDDINGS_DIR, f"{s['id']}.npy"))
        rows.append({
            "Roll No.":   s["roll_number"],
            "Name":       s["name"],
            "Class":      s["class_name"],
            "Section":    s.get("section") or "—",
            "Email":      s["email"],
            "Photos":     s["photos_count"],
            "Status":     "✅ Enrolled" if has_emb else "⚠️ No embedding",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Re-enroll / delete
    st.markdown("---")
    st.markdown("**Update or remove a student**")

    col1, col2 = st.columns(2)
    with col1:
        roll_to_update = st.text_input("Roll number", key="update_roll", placeholder="CS2301")
    with col2:
        action = st.radio("Action", ["Re-enroll (new photos)", "Remove student"], horizontal=True, key="action_radio")

    if st.button("Apply", key="apply_action"):
        student = db.get_student_by_roll(roll_to_update.strip())
        if not student:
            st.error(f"Student with roll **{roll_to_update}** not found")
        elif action == "Remove student":
            photo_path = db.delete_student(student["id"])
            emb_path = os.path.join(EMBEDDINGS_DIR, f"{student['id']}.npy")
            if os.path.exists(emb_path):
                try:
                    os.remove(emb_path)
                except Exception:
                    pass
            if photo_path and os.path.exists(photo_path):
                try:
                    os.remove(photo_path)
                except Exception:
                    pass
            st.success(f"Permanently removed **{student['name']}** ({student['roll_number']}) from the database.")
            st.rerun()
        else:
            st.session_state["reenroll_student"] = student
            st.info(f"Upload new photos for **{student['name']}** below ↓")

    # Re-enrollment upload
    if "reenroll_student" in st.session_state:
        s = st.session_state["reenroll_student"]
        st.markdown(f"**Re-enrolling: {s['name']} ({s['roll_number']})**")
        new_photos = st.file_uploader(
            "New face photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="reenroll_photos",
        )
        if new_photos and st.button("Update Embedding", key="do_reenroll"):
            aligned = []
            for pf in new_photos[:10]:
                img_bgr = fe.bytes_to_bgr(pf.read())
                img_bgr = cv2.resize(img_bgr,None,fx=1.5,fy=1.5)
                img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.2, beta=20)
                faces = fe.detect_faces(img_bgr)
                if faces:
                    faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
                    aligned.append(faces[0].aligned)
            if aligned:
                fe.enroll_student(s["id"], aligned)
                db.update_student_photos(s["id"], len(aligned))
                del st.session_state["reenroll_student"]
                st.success(f"Updated embedding for **{s['name']}** with {len(aligned)} photo(s)")
                st.rerun()
            else:
                st.error("No faces detected in uploaded photos")
