from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import cv2
import numpy as np

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
ARABIC_VARIANTS = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ة": "ه",
    "ى": "ي",
}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"Unreadable or empty image: {p}")
    return img


def normalize_digits(text: str) -> str:
    return text.translate(ARABIC_DIGITS)


def normalize_arabic_text(text: str) -> str:
    text = normalize_digits(text)
    text = text.replace("ـ", "")
    for k, v in ARABIC_VARIANTS.items():
        text = text.replace(k, v)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def only_digits(text: str) -> str:
    return re.sub(r"\D", "", normalize_digits(text))


def validate_egyptian_id(id_digits: str) -> tuple[bool, str]:
    if len(id_digits) != 14:
        return False, "ID must be 14 digits"
    century = id_digits[0]
    if century not in {"2", "3"}:
        return False, "Invalid century digit"
    yy = int(id_digits[1:3])
    mm = int(id_digits[3:5])
    dd = int(id_digits[5:7])
    if not (1 <= mm <= 12):
        return False, "Invalid month"
    if not (1 <= dd <= 31):
        return False, "Invalid day"
    year = (1900 if century == "2" else 2000) + yy
    if not (1900 <= year <= 2099):
        return False, "Year out of range"
    return True, "ok"


def timed(start: float, end: float) -> float:
    return round(float(end - start), 4)
