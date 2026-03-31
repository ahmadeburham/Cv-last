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


def _quad_metrics(quad: np.ndarray, img_shape: tuple[int, int, int], target_ratio: float = 1.5625) -> dict[str, float]:
    h, w = img_shape[:2]
    rect = order_points(quad)
    e1 = float(np.linalg.norm(rect[1] - rect[0]))
    e2 = float(np.linalg.norm(rect[2] - rect[1]))
    e3 = float(np.linalg.norm(rect[3] - rect[2]))
    e4 = float(np.linalg.norm(rect[0] - rect[3]))
    width = (e1 + e3) / 2.0
    height = (e2 + e4) / 2.0
    aspect = width / max(height, 1e-6)
    aspect_score = float(max(0.0, 1.0 - abs(aspect - target_ratio) / target_ratio))

    quad_area = abs(float(cv2.contourArea(rect.astype(np.float32))))
    area_ratio = quad_area / max(float(h * w), 1.0)
    area_score = float(min(1.0, area_ratio / 0.22))

    perimeter = e1 + e2 + e3 + e4
    rectangularity = float((4.0 * np.pi * quad_area) / max(perimeter * perimeter, 1e-6))

    return {
        "aspect_score": aspect_score,
        "area_ratio": area_ratio,
        "area_score": area_score,
        "rectangularity": max(0.0, min(1.0, rectangularity * 1.6)),
    }


def _border_edge_support(gray: np.ndarray, quad: np.ndarray) -> float:
    edges = cv2.Canny(gray, 40, 120)
    rect = order_points(quad).astype(np.int32)
    support_vals: list[float] = []
    for i in range(4):
        p1 = tuple(rect[i])
        p2 = tuple(rect[(i + 1) % 4])
        line_mask = np.zeros_like(gray)
        cv2.line(line_mask, p1, p2, 255, 3)
        line_pixels = float(np.count_nonzero(line_mask))
        if line_pixels <= 0:
            support_vals.append(0.0)
            continue
        edge_hits = float(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=line_mask)))
        support_vals.append(edge_hits / line_pixels)
    return float(np.clip(np.mean(support_vals), 0.0, 1.0))


def _combined_score(
    quad: np.ndarray,
    img: np.ndarray,
    homography_score: float = 0.0,
) -> float:
    m = _quad_metrics(quad, img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    border_score = _border_edge_support(gray, quad)
    combined = (
        0.28 * m["aspect_score"]
        + 0.22 * m["area_score"]
        + 0.22 * m["rectangularity"]
        + 0.18 * border_score
        + 0.10 * float(np.clip(homography_score, 0.0, 1.0))
    )
    return float(np.clip(combined, 0.0, 1.0))


def _contour_candidates(img: np.ndarray, min_area_ratio: float, canny_low: int, canny_high: int) -> list[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(clahe, canny_low, canny_high)
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
        orb = cv2.ORB_create(2400)
        g1 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
            return 0.0, None, "insufficient_features"
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(d1, d2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            return 0.0, None, "insufficient_good_matches"
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        hmat, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.5)
        if hmat is None or mask is None:
            return 0.0, None, "homography_failed"
        inliers = int(mask.ravel().sum())
        h, w = template.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, hmat).reshape(4, 2)
        plausibility = min(1.0, inliers / 90.0)
        return plausibility, proj.astype(np.float32), "ok"
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


def validate_warped_card(warped: np.ndarray) -> tuple[bool, dict[str, float]]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    edges = cv2.Canny(gray, 40, 120)
    text_density = float(np.count_nonzero(edges) / max(edges.size, 1))
    portrait = gray[:, : max(1, gray.shape[1] // 3)]
    portrait_structure = float(cv2.Laplacian(portrait, cv2.CV_64F).var())
    plausible = contrast > 14.0 and text_density > 0.01 and portrait_structure > 8.0
    return plausible, {
        "contrast": contrast,
        "text_density": text_density,
        "portrait_structure": portrait_structure,
    }


def detect_card(
    scene: np.ndarray,
    template: np.ndarray,
    rotations: list[int],
    min_area_ratio: float,
    canny_low: int,
    canny_high: int,
) -> DetectionResult:
    best = DetectionResult(False, 0.0, None, 0, "no_candidate")
    for rot in rotations:
        rimg = rotate_image(scene, rot)
        fscore, fquad, freason = _feature_homography(template, rimg)
        if fquad is not None:
            conf = _combined_score(fquad, rimg, homography_score=fscore)
            if conf > best.confidence:
                best = DetectionResult(True, conf, fquad, rot, "feature_homography")
        quads = _contour_candidates(rimg, min_area_ratio, canny_low, canny_high)
        for q in quads:
            conf = _combined_score(q, rimg, homography_score=0.0)
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
