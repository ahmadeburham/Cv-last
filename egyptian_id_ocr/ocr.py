from __future__ import annotations

from dataclasses import dataclass
import re
import cv2
import numpy as np

from .utils import normalize_arabic_text, only_digits, validate_egyptian_id


@dataclass
class OCRResult:
    text: str
    backend: str
    confidence: float
    variant: str
    score: float


_EASYOCR_READER = None


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
        morph = cv2.morphologyEx(resized, cv2.MORPH_BLACKHAT, kernel)
        _, mbin = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["digit_blackhat"] = mbin
    else:
        blur = cv2.GaussianBlur(resized, (0, 0), 1.0)
        sharp = cv2.addWeighted(resized, 1.7, blur, -0.7, 0)
        variants["text_sharp"] = sharp
    return variants


def _tess_config_for_field(field_name: str) -> str:
    if field_name in {"id_number", "birth_date"}:
        return "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789٠١٢٣٤٥٦٧٨٩/.-"
    return "--oem 3 --psm 6"


def _ocr_pytesseract(img: np.ndarray, lang: str, field_name: str) -> tuple[str, float]:
    import pytesseract

    config = _tess_config_for_field(field_name)
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


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr

        _EASYOCR_READER = easyocr.Reader(["ar", "en"], gpu=False, verbose=False)
    return _EASYOCR_READER


def _ocr_easyocr(img: np.ndarray) -> tuple[str, float]:
    reader = _get_easyocr_reader()
    res = reader.readtext(img)
    text = " ".join([r[1] for r in res]).strip()
    conf = float(np.mean([r[2] for r in res])) if res else 0.0
    return text, conf


def _garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    allowed = re.findall(r"[\w\s\u0600-\u06FF\d]", text)
    return float(max(0.0, 1.0 - len(allowed) / max(1, len(text))))


def _score_field_text(field_name: str, text: str, conf: float) -> float:
    raw = text.strip()
    if not raw:
        return 0.0

    if field_name == "id_number":
        digits = only_digits(raw)
        plaus, _ = validate_egyptian_id(digits)
        digit_clean = len(digits) / max(1, len(raw))
        len_score = max(0.0, 1.0 - abs(len(digits) - 14) / 14.0)
        noise_penalty = _garbage_ratio(raw)
        return float(0.45 * digit_clean + 0.3 * len_score + 0.2 * (1.0 if plaus else 0.0) + 0.1 * conf - 0.25 * noise_penalty)

    if field_name == "birth_date":
        digits = only_digits(raw)
        digit_clean = len(digits) / max(1, len(raw))
        date_like = 1.0 if len(digits) in {6, 8} else 0.0
        noise_penalty = _garbage_ratio(raw)
        return float(0.5 * digit_clean + 0.25 * date_like + 0.1 * conf - 0.25 * noise_penalty)

    norm = normalize_arabic_text(raw)
    words = [w for w in norm.split(" ") if w]
    multi_word = min(1.0, len(words) / 3.0)
    noise_penalty = _garbage_ratio(raw)
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", norm)) / max(1, len(norm))
    return float(0.45 * arabic_chars + 0.25 * multi_word + 0.1 * conf - 0.25 * noise_penalty)


def run_ocr_with_fallback(
    field_name: str,
    variants: dict[str, np.ndarray],
    primary: str,
    fallbacks: list[str],
    tesseract_lang: str,
) -> OCRResult:
    backends = [primary] + [b for b in fallbacks if b != primary]
    best = OCRResult("", "none", 0.0, "none", 0.0)
    for vname, img in variants.items():
        for backend in backends:
            try:
                if backend == "pytesseract":
                    txt, conf = _ocr_pytesseract(img, tesseract_lang, field_name)
                elif backend == "easyocr":
                    txt, conf = _ocr_easyocr(img)
                else:
                    continue
                txt = txt.strip()
                score = _score_field_text(field_name, txt, conf)
                if txt and score >= best.score:
                    best = OCRResult(txt, backend, conf, vname, score)
            except Exception:
                continue
    return best
