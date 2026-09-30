

import os
import uuid
import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import streamlit as st

import insightface
from insightface.app import FaceAnalysis

from config import EMBEDDINGS_DIR, FACES_DIR, MATCH_THRESHOLD, MIN_SCORE, FACE_MODEL

logger = logging.getLogger(__name__)


# ── Detection constants ───────────────────────────────────────────────────────

# Lower threshold for classroom (many small/partial faces); stricter for enrollment
MIN_SCORE_CLASSROOM  = max(0.30, MIN_SCORE - 0.20)
MIN_SCORE_ENROLLMENT = max(0.60, MIN_SCORE)

# Tile overlap fraction (0.25 = 25 % overlap between adjacent tiles)
TILE_OVERLAP = 0.25

# NMS IoU threshold — detections with IoU above this are merged
NMS_IOU_THRESH = 0.45

# Max side length sent to the detector in one shot.
# InsightFace det_size must be a multiple of 32.
MAX_DET_SIDE = 1280


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DetectedFace:
    index: int
    bbox: tuple          # (x, y, w, h) in original image coordinates
    score: float
    aligned: np.ndarray  # 112×112 BGR, ready for ArcFace
    crop: np.ndarray     # padded crop for display


@dataclass
class MatchResult:
    student_id: Optional[str]
    student_name: Optional[str]
    confidence: float        # 0–1, higher = better
    distance: float          # cosine distance, lower = better
    is_known: bool


@dataclass
class ClassroomResult:
    session_id: str
    faces: List[DetectedFace]
    matches: List[MatchResult]
    present_ids: set
    unknown_count: int
    annotated_image: np.ndarray


# ── ArcFace canonical landmarks for 112×112 alignment ─────────────────────────

_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading face recognition models…")
def load_models():
    """
    Load InsightFace models once.
    Compatible with insightface 0.6.x and 0.7.x.
    buffalo_l downloads ~300 MB on first run; subsequent runs use local cache.
    """
    logger.info(f"Loading InsightFace model pack: {FACE_MODEL}")

    try:
        app = FaceAnalysis(
            name=FACE_MODEL,
            providers=["CPUExecutionProvider"],
        )
    except TypeError:
        app = FaceAnalysis(name=FACE_MODEL)

    # We set det_size dynamically per call — use a sensible default here.
    app.prepare(ctx_id=0, det_size=(640, 640))

    rec_model = None
    for key in ("recognition", "rec", "arcface"):
        rec_model = app.models.get(key)
        if rec_model is not None:
            break

    if rec_model is None:
        for key, model in app.models.items():
            if "det" not in key.lower():
                rec_model = model
                logger.warning(f"Using fallback recognition model key: '{key}'")
                break

    if rec_model is None:
        raise RuntimeError(
            "Could not find recognition model inside FaceAnalysis. "
            "Check your insightface installation and model pack."
        )

    logger.info("Models loaded successfully")
    return app, rec_model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_for_detection(image_bgr: np.ndarray) -> np.ndarray:
    """
    Improve detection in challenging classroom conditions:
      1. CLAHE on the L channel (YCrCb) to fix uneven / dim lighting.
      2. Mild sharpening to recover soft edges from camera blur.

    Operates in-place on a copy — original is not modified.
    """
    img = image_bgr.copy()

    # ── CLAHE lightness equalisation ──────────────────────────────────────────
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    y = clahe.apply(y)

    img = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    # ── Unsharp mask (mild sharpening) ────────────────────────────────────────
    blur   = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    img    = cv2.addWeighted(img, 1.4, blur, -0.4, 0)

    return img


# ── Adaptive det_size ─────────────────────────────────────────────────────────

def _best_det_size(h: int, w: int) -> Tuple[int, int]:
    """
    Choose the detector input resolution that best covers the image
    while staying ≤ MAX_DET_SIDE and aligned to 32 px.
    Larger images get a higher det_size so small distant faces are not missed.
    """
    longest = max(h, w)
    # Scale up to MAX_DET_SIDE if the image is large
    target = min(longest, MAX_DET_SIDE)
    # Snap to nearest multiple of 32 (required by InsightFace ONNX models)
    target = max(32, (target // 32) * 32)
    return (target, target)


# ── Tiled detection ───────────────────────────────────────────────────────────

def _detect_on_patch(
    app: FaceAnalysis,
    patch: np.ndarray,
    det_size: Tuple[int, int],
    min_score: float,
    offset_x: int,
    offset_y: int,
    scale: float,
) -> List[dict]:
    """
    Run InsightFace on a single patch and return raw detections translated
    back into original-image coordinates.
    """
    # Temporarily reprepare with the patch det_size if needed
    app.prepare(ctx_id=0, det_size=det_size)
    raw = app.get(patch)

    faces = []
    for face in raw:
        if face.det_score < min_score:
            continue

        x1, y1, x2, y2 = face.bbox
        # Map back to original coordinate space
        x1 = x1 / scale + offset_x
        y1 = y1 / scale + offset_y
        x2 = x2 / scale + offset_x
        y2 = y2 / scale + offset_y

        kps = face.kps.copy() / scale
        kps[:, 0] += offset_x
        kps[:, 1] += offset_y

        faces.append({
            "bbox"  : [x1, y1, x2, y2],
            "score" : float(face.det_score),
            "kps"   : kps,
        })
    return faces


def _iou(a: List[float], b: List[float]) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: List[dict], iou_thresh: float = NMS_IOU_THRESH) -> List[dict]:
    """
    Score-weighted NMS.
    Keeps the highest-score box from any group of overlapping detections.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    suppressed = [False] * len(dets)

    for i in range(len(dets)):
        if suppressed[i]:
            continue
        kept.append(dets[i])
        for j in range(i + 1, len(dets)):
            if not suppressed[j]:
                if _iou(dets[i]["bbox"], dets[j]["bbox"]) >= iou_thresh:
                    suppressed[j] = True

    return kept


def _tile_and_detect(
    image_bgr: np.ndarray,
    app: FaceAnalysis,
    min_score: float,
) -> List[dict]:
    """
    Multi-scale tiled detection strategy:

    Pass 1 — Full image rescaled to MAX_DET_SIDE.
              Catches large / close-up faces.

    Pass 2 — If the image is wide enough (width > 1.5× height),
              split into overlapping horizontal tiles and run again.
              Catches small faces in wide classroom panoramas.

    All detections are merged and de-duplicated with NMS.
    """
    H, W = image_bgr.shape[:2]
    all_dets: List[dict] = []

    # ── Pass 1: full image ────────────────────────────────────────────────────
    det_size = _best_det_size(H, W)
    scale    = min(det_size[0] / H, det_size[1] / W)
    resized  = cv2.resize(image_bgr, (int(W * scale), int(H * scale)))

    dets = _detect_on_patch(app, resized, det_size, min_score, 0, 0, scale)
    all_dets.extend(dets)
    logger.debug(f"Pass 1 (full image @{det_size}): {len(dets)} raw faces")

    # ── Pass 2: horizontal tiling for wide images ─────────────────────────────
    if W > 1.5 * H and W > 1000:
        n_tiles  = max(2, round(W / H))
        step     = W / n_tiles
        overlap  = int(step * TILE_OVERLAP)
        tile_w   = int(step) + overlap

        for t in range(n_tiles):
            tx1 = max(0, int(t * step) - overlap // 2)
            tx2 = min(W, tx1 + tile_w)
            tile = image_bgr[:, tx1:tx2]

            t_det_size = _best_det_size(H, tx2 - tx1)
            t_scale    = min(t_det_size[0] / H, t_det_size[1] / (tx2 - tx1))
            t_resized  = cv2.resize(tile, (int((tx2-tx1)*t_scale), int(H*t_scale)))

            tile_dets = _detect_on_patch(
                app, t_resized, t_det_size, min_score,
                offset_x=tx1, offset_y=0, scale=t_scale
            )
            all_dets.extend(tile_dets)
            logger.debug(f"Pass 2 tile {t}: {len(tile_dets)} raw faces")

    # ── Pass 3: vertical tiling for tall images ───────────────────────────────
    if H > 1.5 * W and H > 1000:
        n_tiles  = max(2, round(H / W))
        step     = H / n_tiles
        overlap  = int(step * TILE_OVERLAP)
        tile_h   = int(step) + overlap

        for t in range(n_tiles):
            ty1 = max(0, int(t * step) - overlap // 2)
            ty2 = min(H, ty1 + tile_h)
            tile = image_bgr[ty1:ty2, :]

            t_det_size = _best_det_size(ty2 - ty1, W)
            t_scale    = min(t_det_size[0] / (ty2-ty1), t_det_size[1] / W)
            t_resized  = cv2.resize(tile, (int(W*t_scale), int((ty2-ty1)*t_scale)))

            tile_dets = _detect_on_patch(
                app, t_resized, t_det_size, min_score,
                offset_x=0, offset_y=ty1, scale=t_scale
            )
            all_dets.extend(tile_dets)
            logger.debug(f"Pass 3 tile {t}: {len(tile_dets)} raw faces")

    final = _nms(all_dets)
    logger.info(f"After NMS: {len(final)} faces (from {len(all_dets)} raw detections)")
    return final


# ── Public API ────────────────────────────────────────────────────────────────

def detect_faces(
    image_bgr: np.ndarray,
    min_score: float = MIN_SCORE_CLASSROOM,
    preprocess: bool = True,
) -> List[DetectedFace]:
    """
    Detect all faces in an image using RetinaFace with multi-scale tiled detection.

    Args:
        image_bgr:   BGR image (any size).
        min_score:   Minimum RetinaFace confidence to keep a detection.
                     Defaults to MIN_SCORE_CLASSROOM (lower = catch more faces).
        preprocess:  Apply CLAHE + sharpening before detection (recommended).

    Returns:
        List of DetectedFace, sorted left → right by x position.
    """
    app, _ = load_models()

    work = _preprocess_for_detection(image_bgr) if preprocess else image_bgr.copy()
    raw_dets = _tile_and_detect(work, app, min_score)

    H, W = image_bgr.shape[:2]
    results: List[DetectedFace] = []

    for i, det in enumerate(raw_dets):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        # Clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Padded crop for display (10 % padding on each side)
        pad_x = max(4, int((x2 - x1) * 0.10))
        pad_y = max(4, int((y2 - y1) * 0.10))
        cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
        cx2 = min(W, x2 + pad_x); cy2 = min(H, y2 + pad_y)
        crop = image_bgr[cy1:cy2, cx1:cx2].copy()

        # Alignment uses the preprocessed image's landmarks mapped back to
        # original coordinates (we stored kps already in original space)
        aligned = _align_face(image_bgr, det["kps"])

        results.append(DetectedFace(
            index=i,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            score=float(det["score"]),
            aligned=aligned,
            crop=crop,
        ))

    results.sort(key=lambda f: f.bbox[0])
    return results


def get_embedding(aligned_112: np.ndarray) -> np.ndarray:
    """
    Generate 512-dim L2-normalised ArcFace embedding from a 112×112 aligned face.
    Compatible with insightface 0.6.x and 0.7.x.
    """
    _, rec_model = load_models()
    face_input = aligned_112.astype(np.float32)

    if hasattr(rec_model, "get_feat"):
        emb = rec_model.get_feat(face_input).flatten()
    else:
        blob = cv2.dnn.blobFromImage(
            aligned_112, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        rec_model.input_size = (112, 112)
        emb = rec_model.get(aligned_112).flatten()

    norm = np.linalg.norm(emb)
    return (emb / norm).astype(np.float32) if norm > 0 else emb.astype(np.float32)


def enroll_student(
    student_id: str,
    aligned_faces: List[np.ndarray],
    augment: bool = True,
) -> np.ndarray:
    """
    Compute enrollment embedding from 1–N aligned face images.

    Args:
        student_id:    Unique student identifier (used as filename).
        aligned_faces: List of 112×112 aligned BGR face images.
        augment:       If True, include horizontally flipped versions to make
                       the embedding more robust to pose variation.

    Strategy:
        - Average individual L2-normalised embeddings, then re-normalise.
        - Optional horizontal flip augmentation doubles the sample count.
        - More photos + augmentation = more robust embedding.
    """
    if not aligned_faces:
        raise ValueError("Need at least 1 face image")

    images = list(aligned_faces)

    # Horizontal flip augmentation — cheap and effective
    if augment:
        images += [cv2.flip(f, 1) for f in aligned_faces]

    embeddings = [get_embedding(f) for f in images]
    avg = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    avg = (avg / norm).astype(np.float32) if norm > 0 else avg.astype(np.float32)

    path = os.path.join(EMBEDDINGS_DIR, f"{student_id}.npy")
    np.save(path, avg)
    return avg


def load_all_embeddings(class_students: List[dict]) -> Dict[str, Tuple[str, np.ndarray]]:
    """
    Load embeddings for all students in a class into memory.
    Returns { student_id: (student_name, embedding_array) }
    """
    db = {}
    for s in class_students:
        path = os.path.join(EMBEDDINGS_DIR, f"{s['id']}.npy")
        if os.path.exists(path):
            emb = np.load(path).astype(np.float32)
            db[s["id"]] = (s["name"], emb)
    return db


def identify_face(
    query_emb: np.ndarray,
    db: Dict[str, Tuple[str, np.ndarray]],
) -> MatchResult:
    """
    1:N cosine distance match — find who this face belongs to.
    Cosine distance = 1 - dot_product  (both sides are L2-normalised).
    """
    if not db:
        return MatchResult(None, None, 0.0, 1.0, False)

    ids    = list(db.keys())
    names  = [db[i][0] for i in ids]
    matrix = np.stack([db[i][1] for i in ids])

    distances = 1.0 - (matrix @ query_emb)
    best_idx  = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    confidence = round(max(0.0, 1.0 - best_dist), 4)

    if best_dist <= MATCH_THRESHOLD:
        return MatchResult(
            student_id=ids[best_idx],
            student_name=names[best_idx],
            confidence=confidence,
            distance=round(best_dist, 4),
            is_known=True,
        )
    return MatchResult(None, None, confidence, round(best_dist, 4), False)


def identify_batch(
    query_embs: List[np.ndarray],
    db: Dict[str, Tuple[str, np.ndarray]],
) -> List[MatchResult]:
    """
    Vectorised batch 1:N matching — one matrix multiply for all faces.
    Much faster than calling identify_face() in a loop.
    """
    if not db or not query_embs:
        return [MatchResult(None, None, 0.0, 1.0, False)] * len(query_embs)

    ids    = list(db.keys())
    names  = [db[i][0] for i in ids]
    matrix = np.stack([db[i][1] for i in ids])   # (M, 512)
    Q      = np.stack(query_embs)                 # (N, 512)
    D      = 1.0 - (Q @ matrix.T)                # (N, M) distance matrix

    results = []
    for row in D:
        best_idx  = int(np.argmin(row))
        best_dist = float(row[best_idx])
        conf      = round(max(0.0, 1.0 - best_dist), 4)
        if best_dist <= MATCH_THRESHOLD:
            results.append(MatchResult(ids[best_idx], names[best_idx], conf, round(best_dist, 4), True))
        else:
            results.append(MatchResult(None, None, conf, round(best_dist, 4), False))

    return results


def run_classroom_pipeline(
    image_bgr: np.ndarray,
    class_students: List[dict],
) -> ClassroomResult:
    """
    Full pipeline:
    1. Preprocess + detect all faces (multi-scale tiled RetinaFace)
    2. Generate ArcFace embeddings
    3. Batch 1:N match against class student embeddings
    4. Return annotated image + attendance results
    """
    session_id = str(uuid.uuid4())

    # Step 1 — detect (uses classroom threshold, lower = catch more faces)
    faces = detect_faces(image_bgr, min_score=MIN_SCORE_CLASSROOM, preprocess=True)

    # Step 2 — load DB + embed + match
    db = load_all_embeddings(class_students)
    present_ids: set = set()
    matches: List[MatchResult] = []

    if faces and db:
        embeddings = []
        valid_faces = []
        for face in faces:
            try:
                embeddings.append(get_embedding(face.aligned))
                valid_faces.append(face)
            except Exception as e:
                logger.warning(f"Embedding failed for face {face.index}: {e}")

        matches = identify_batch(embeddings, db)

        for match in matches:
            if match.is_known and match.student_id:
                present_ids.add(match.student_id)

    unknown_count = sum(1 for m in matches if not m.is_known)

    # Step 3 — annotate
    annotated = _draw_results(image_bgr.copy(), faces, matches)

    # Step 4 — save face crops
    for i, face in enumerate(faces):
        crop_path = os.path.join(FACES_DIR, f"{session_id}_face{i}.jpg")
        cv2.imwrite(crop_path, face.crop)

    return ClassroomResult(
        session_id=session_id,
        faces=faces,
        matches=matches,
        present_ids=present_ids,
        unknown_count=unknown_count,
        annotated_image=annotated,
    )


# ── Utilities ──────────────────────────────────────────────────────────────────

def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def pil_to_bgr(pil_image) -> np.ndarray:
    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _align_face(img_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Align face to ArcFace canonical 112×112 using 5-point landmarks."""
    src = landmarks.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        x1 = max(0, int(src[:, 0].min()))
        y1 = max(0, int(src[:, 1].min()))
        x2 = min(img_bgr.shape[1], int(src[:, 0].max()))
        y2 = min(img_bgr.shape[0], int(src[:, 1].max()))
        return cv2.resize(img_bgr[y1:y2, x1:x2], (112, 112))
    return cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0)


def _draw_results(
    img: np.ndarray,
    faces: List[DetectedFace],
    matches: List[MatchResult],
) -> np.ndarray:
    """Draw bounding boxes + names on the classroom image."""
    for face, match in zip(faces, matches):
        x, y, w, h = face.bbox
        if match.is_known:
            color = (0, 200, 80)
            label = f"{match.student_name}  {match.confidence*100:.0f}%"
        else:
            color = (0, 80, 220)
            label = f"Unknown  ({match.distance:.2f})"

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(img, label, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, "SmartAttend", (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return img
