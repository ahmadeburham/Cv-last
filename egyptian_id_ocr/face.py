from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FaceMatchResult:
    status: str
    verified: bool | None
    score: float | None
    threshold: float
    reason: str


def detect_face_crop(img: np.ndarray, min_size: int = 48) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None, None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return img[y : y + h, x : x + w].copy(), (int(x), int(y), int(w), int(h))


def _embedding_face_recognition(face: np.ndarray) -> np.ndarray | None:
    import face_recognition

    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)
    if not enc:
        return None
    return enc[0]


def _fallback_embedding(face: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([resized], [0], None, [64], [0, 256]).flatten()
    hist /= (np.linalg.norm(hist) + 1e-8)
    return hist


def compare_faces(id_face: np.ndarray, selfie_face: np.ndarray, threshold: float) -> FaceMatchResult:
    try:
        e1 = _embedding_face_recognition(id_face)
        e2 = _embedding_face_recognition(selfie_face)
        if e1 is None or e2 is None:
            return FaceMatchResult("failed", None, None, threshold, "face_recognition_no_embedding")
        dist = float(np.linalg.norm(e1 - e2))
        score = 1.0 - dist
        return FaceMatchResult("success", score >= threshold, score, threshold, "face_recognition")
    except Exception:
        e1 = _fallback_embedding(id_face)
        e2 = _fallback_embedding(selfie_face)
        score = float(np.dot(e1, e2))
        return FaceMatchResult("success", score >= threshold, score, threshold, "fallback_histogram")
