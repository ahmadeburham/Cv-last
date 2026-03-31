from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class OCRResult:
    text: str
    backend: str
    confidence: float
    variant: str


def preprocess_variants(field_name: str, img: np.ndarray, scale_factor: float = 2.0) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {}
    if img is None or img.size == 0:
        img = np.zeros((32, 128, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    variants["gray_resized"] = resized

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    variants["clahe"] = clahe

    med = cv2.medianBlur(clahe, 3)
    variants["median"] = med

    _, otsu = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["otsu"] = otsu

    adapt = cv2.adaptiveThreshold(med, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    variants["adaptive"] = adapt

    if field_name in {"id_number", "birth_date"}:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blackhat = cv2.morphologyEx(resized, cv2.MORPH_BLACKHAT, kernel)
        _, bh_bin = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["digit_blackhat"] = bh_bin
    else:
        blur = cv2.GaussianBlur(resized, (0, 0), 1.0)
        sharp = cv2.addWeighted(resized, 1.6, blur, -0.6, 0)
        variants["text_sharp"] = sharp

    return variants


def _ocr_pytesseract(img: np.ndarray, lang: str, psm: int = 6) -> tuple[str, float]:
    import pytesseract

    config = f"--oem 3 --psm {psm}"
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    conf = 0.0
    try:
        data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
        vals = [float(x) for x in data.get("conf", []) if x not in {"-1", -1}]
        if vals:
            conf = float(np.mean(vals) / 100.0)
    except Exception:
        conf = 0.0
    return text.strip(), conf


def _ocr_easyocr(img: np.ndarray) -> tuple[str, float]:
    import easyocr

    reader = easyocr.Reader(["ar", "en"], gpu=False, verbose=False)
    res = reader.readtext(img)
    text = " ".join([r[1] for r in res]).strip()
    conf = float(np.mean([r[2] for r in res])) if res else 0.0
    return text, conf


def run_ocr_with_fallback(
    variants: dict[str, np.ndarray],
    primary: str,
    fallbacks: list[str],
    tesseract_lang: str,
) -> OCRResult:
    backends = [primary] + [b for b in fallbacks if b != primary]
    best = OCRResult("", "none", 0.0, "none")
    for vname, img in variants.items():
        for backend in backends:
            try:
                if backend == "pytesseract":
                    txt, conf = _ocr_pytesseract(img, tesseract_lang)
                elif backend == "easyocr":
                    txt, conf = _ocr_easyocr(img)
                else:
                    continue
                txt = txt.strip()
                if txt and conf >= best.confidence:
                    best = OCRResult(txt, backend, conf, vname)
            except Exception:
                continue
    return best
