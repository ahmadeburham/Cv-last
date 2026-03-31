from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class FieldBox:
    x: float
    y: float
    w: float
    h: float


@dataclass
class OCRConfig:
    primary_backend: str = "pytesseract"
    fallback_backends: list[str] = field(default_factory=lambda: ["easyocr"])
    tesseract_lang: str = "ara+eng"
    min_text_len: int = 1


@dataclass
class FaceConfig:
    match_threshold: float = 0.55
    min_face_size: int = 48


@dataclass
class DetectionConfig:
    rotation_candidates: list[int] = field(default_factory=lambda: [0, 90, 180, 270])
    canny_low: int = 50
    canny_high: int = 180
    min_card_area_ratio: float = 0.08
    min_confidence: float = 0.15


@dataclass
class PreprocessConfig:
    scale_factor: float = 2.0
    apply_clahe: bool = True
    median_blur_ksize: int = 3


@dataclass
class PipelineConfig:
    template_path: str = ""
    output_dir: str = "artifacts/runs"
    debug: bool = True
    aligned_size: tuple[int, int] = (1000, 640)
    fields: dict[str, FieldBox] = field(default_factory=dict)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    @staticmethod
    def from_json(path: str | Path) -> "PipelineConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return from_dict(data)


def from_dict(data: dict[str, Any]) -> PipelineConfig:
    fields = {
        k: FieldBox(**v)
        for k, v in data.get("fields", {}).items()
    }
    return PipelineConfig(
        template_path=data.get("template_path", ""),
        output_dir=data.get("output_dir", "artifacts/runs"),
        debug=bool(data.get("debug", True)),
        aligned_size=tuple(data.get("aligned_size", [1000, 640])),
        fields=fields,
        ocr=OCRConfig(**data.get("ocr", {})),
        face=FaceConfig(**data.get("face", {})),
        detection=DetectionConfig(**data.get("detection", {})),
        preprocess=PreprocessConfig(**data.get("preprocess", {})),
    )
