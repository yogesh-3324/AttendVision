# AttendVision — AI Face Recognition Attendance System

Full pipeline: **Classroom photo → RetinaFace detection → Face alignment → ArcFace embedding → 1:N cosine matching → Mark attendance → Send emails**

Built with **Streamlit** + **InsightFace** (RetinaFace + ArcFace).

---

## Setup (5 minutes)

### 1. Install dependencies
```bash
python -m venv venv
.\venv\Scripts\activate   # Windows (or source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```
> First run downloads the InsightFace model (~300 MB). Cached locally after that.

### 2. Configure email & environment
```bash
cp .env.example .env
# Edit .env with your Gmail App Password or SMTP credentials
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
Opens in your browser at **http://localhost:8501**

---

## How it works

### Step 1 — Enroll Students
- Navigate to **👤 Enroll Students**
- Enter student details (Name, Roll Number, Class, Email, Section, Phone) + upload 3–5 clear face photos
- System runs **RetinaFace** to extract faces and **ArcFace** to compute 512-dim normalized embeddings
- Embeddings are saved to `data/embeddings/{student_id}.npy`

### Step 2 — Take Attendance
- Navigate to **📷 Take Attendance**
- Select class, subject, date, and upload a classroom photo
- The pipeline executes automatically:
  1. **RetinaFace** detects all faces in the image (handles occlusions and angles)
  2. **Face alignment** to canonical 112×112 using 5 facial landmarks
  3. **ArcFace** generates a 512-dim embedding per detected face
  4. **1:N cosine distance search** matches each face against all enrolled student embeddings
  5. Students not detected in the image are marked **Absent**
- Visual feedback: View annotated image with green bounding boxes (Present) and red (Unknown), plus confidence scores
- Save attendance and optionally send instant automated email notifications

### Step 3 — Records & Reports
- **📋 Attendance Records**: Browse past lecture records, review detection rates, and inspect session details
- **📊 Reports & Analytics**: View weekly attendance trends, rate fluctuations, per-student attendance rates, and automatic low attendance alerts (<75%)

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
  → each face becomes a point in 512-dim hypersphere
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
| RetinaFace for detection | Robust multi-face detection handling occlusions, angles, and scale |
| Average of N enrollment embeddings | More robust feature vector than single-photo enrollment |
| Cosine distance, not Euclidean | L2-normalized vectors — cosine angle is the correct metric |
| Vectorized 1:N matching | Single matrix multiplication for all faces — sub-second matching |
| 0.4 threshold (configurable) | Optimal balance between precision and recall |
| `.npy` files for embeddings | Instant disk-to-memory vector arrays without DB BLOB overhead |
| SQLite | Zero-config, reliable relational persistence |

---

## Project Structure

```
AttendVision/
├── app.py              # Streamlit dashboard entry point
├── config.py           # Configuration and environment settings
├── face_engine.py      # Core AI: RetinaFace + ArcFace pipeline
├── database.py         # SQLite persistence and data layer
├── email_service.py    # SMTP email dispatch & HTML alerts
├── requirements.txt    # Project dependencies
├── .env.example        # Environment variable template
├── pages/
│   ├── capture.py      # Take Attendance page
│   ├── enroll.py       # Enroll Students page
│   ├── records.py      # Attendance Records page
│   └── reports.py      # Reports & Analytics page
└── data/
    ├── embeddings/     # .npy embedding files per student
    ├── uploads/        # Classroom & profile photos
    ├── faces/          # Cropped face thumbnails
    └── attendvision.db # SQLite database
```
