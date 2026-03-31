from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class DetectionResult:
    found: bool
    confidence: float
    quad: np.ndarray | None
    rotation: int
    reason: str


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _aspect_score(quad: np.ndarray, target_ratio: float = 1.5625) -> float:
    tl, tr, br, bl = order_points(quad)
    w = np.linalg.norm(tr - tl)
    h = np.linalg.norm(bl - tl)
    if h <= 0:
        return 0.0
    ratio = w / h
    return float(max(0.0, 1.0 - abs(ratio - target_ratio) / target_ratio))


def _contour_candidates(img: np.ndarray, min_area_ratio: float) -> list[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(clahe, 50, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    min_area = h * w * min_area_ratio
    quads: list[np.ndarray] = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area >= min_area:
                quads.append(approx.reshape(4, 2).astype(np.float32))
    return quads


def _feature_homography(template: np.ndarray, img: np.ndarray) -> tuple[float, np.ndarray | None, str]:
    try:
        orb = cv2.ORB_create(2000)
        g1 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return 0.0, None, "insufficient_features"
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(d1, d2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return 0.0, None, "insufficient_good_matches"
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        hmat, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if hmat is None or mask is None:
            return 0.0, None, "homography_failed"
        inliers = int(mask.ravel().sum())
        h, w = template.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, hmat).reshape(4, 2)
        return min(1.0, inliers / 80.0), proj.astype(np.float32), "ok"
    except Exception as exc:
        return 0.0, None, f"feature_error:{exc}"


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img.copy()
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img.copy()


def detect_card(scene: np.ndarray, template: np.ndarray, rotations: list[int], min_area_ratio: float) -> DetectionResult:
    best = DetectionResult(False, 0.0, None, 0, "no_candidate")
    for rot in rotations:
        rimg = rotate_image(scene, rot)
        fscore, fquad, freason = _feature_homography(template, rimg)
        if fquad is not None:
            ascore = _aspect_score(fquad)
            conf = 0.7 * fscore + 0.3 * ascore
            if conf > best.confidence:
                best = DetectionResult(True, conf, fquad, rot, "feature_homography")
        quads = _contour_candidates(rimg, min_area_ratio)
        for q in quads:
            ascore = _aspect_score(q)
            conf = 0.55 * ascore + 0.2
            if conf > best.confidence:
                best = DetectionResult(True, conf, q, rot, "contour_quad")
        if not best.found:
            best.reason = freason
    return best


def warp_from_quad(img: np.ndarray, quad: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    rect = order_points(quad)
    w, h = out_size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, mat, (w, h))
