# SmartAttend — Face Recognition Attendance System

Full pipeline: **Classroom photo → RetinaFace detection → Face alignment → ArcFace embedding → 1:N cosine matching → Mark attendance → Send emails**

Built with Streamlit + InsightFace (RetinaFace + ArcFace).

---

## Setup (5 minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads the InsightFace model (~300 MB). Cached after that.

### 2. Configure email (optional)
```bash
cp .env.example .env
# Edit .env with your Gmail App Password or SMTP credentials
```

### 3. Run
```bash
streamlit run app.py
```
Opens at http://localhost:8501

---

## How it works

### Step 1 — Enroll Students
- Go to **Enroll Students**
- Enter student details + upload 3–5 clear face photos
- System runs RetinaFace to extract faces, ArcFace to compute 512-dim embeddings
- Embeddings saved to `data/embeddings/{student_id}.npy`

### Step 2 — Take Attendance
- Go to **Take Attendance**
- Select class, upload classroom photo
- Pipeline runs automatically:
  1. **RetinaFace** detects all faces in the image
  2. Face alignment to canonical 112×112 (critical for accuracy)
  3. **ArcFace** generates 512-dim embedding per face
  4. **1:N cosine search** matches each face against all student embeddings
  5. Students not detected → marked absent
- See annotated image with names + confidence scores
- Save attendance and/or send emails

### Step 3 — Report
- View weekly trends, per-student history
- Export CSV
- Low attendance alerts

---

## Architecture

```
Classroom Image
      │
      ▼
RetinaFace Detection (buffalo_l)
  → bounding boxes + 5-point landmarks
      │
      ▼
Face Alignment (affine warp → 112×112)
  → consistent pose regardless of head tilt
      │
      ▼
ArcFace Embedding (512-dim vector, L2-normalised)
  → each face becomes a point in 512-dim space
      │
      ▼
1:N Cosine Distance Search
  → matrix multiply: (N_faces, 512) × (N_students, 512)ᵀ
  → find minimum distance per face
  → threshold: dist < 0.4 → PRESENT, else → UNKNOWN
      │
      ▼
Attendance Records (SQLite)
      │
      ▼
Email Notifications (SMTP)
```

## Key design decisions

| Decision | Why |
|---|---|
| ArcFace over FaceNet/DeepFace | State-of-the-art accuracy on LFW benchmark (99.83%) |
| RetinaFace for detection | Handles occlusion, angles, small faces better than MTCNN |
| Average of N enrollment embeddings | More robust than single-photo enrollment |
| Cosine distance, not Euclidean | L2-normalised vectors — cosine is the right metric |
| Vectorised 1:N matching | One matrix multiply for all faces — fast at scale |
| 0.4 threshold (configurable) | Balanced precision/recall — tune in .env |
| .npy files for embeddings | Faster than DB BLOB, easy to update per student |
| SQLite | Zero-config, perfect for single-server deployment |


## Project structure

```
smartattend/
├── app.py              # Streamlit entry point
├── config.py           # All settings
├── face_engine.py      # Core: RetinaFace + ArcFace pipeline
├── database.py         # SQLite persistence
├── email_service.py    # SMTP email dispatch
├── requirements.txt
├── .env.example
├── pages/
│   ├── capture.py      # Take Attendance page
│   ├── enroll.py       # Enroll Students page
│   ├── records.py      # Attendance Records page
│   └── reports.py      # Reports & Analytics page
└── data/
    ├── embeddings/     # .npy embedding files per student
    ├── uploads/        # Classroom + profile photos
    ├── faces/          # Cropped face thumbnails
    └── smartattend.db  # SQLite database
```
