# Final Change Log

## Modified files
- `egyptian_id_ocr/detection.py`
  - Added config-driven Canny thresholds.
  - Added richer candidate scoring metrics (aspect, area, rectangularity, border support, homography plausibility).
  - Added post-warp sanity validator.
- `egyptian_id_ocr/ocr.py`
  - Added field-specific Tesseract modes.
  - Added field-aware OCR candidate selection scoring.
  - Added EasyOCR reader cache.
- `egyptian_id_ocr/face.py`
  - Changed histogram-only fallback semantics to `weak_fallback` with `verified: null`.
- `egyptian_id_ocr/pipeline.py`
  - Added role classification and scene-only evaluation filtering.
  - Replaced brightness-only orientation with multi-signal scorer.
  - Added crop validation warnings and warp sanity rejection.
  - Added honest status derivation (`success`/`partial`/`failed`).
  - Added guaranteed final result write path for partial failures.
- `run_pipeline.py`
  - Enforced canonical template filename `main_temp.jpeg`.
  - Added explicit input role handling in fallback path and scene-only benchmark counting.
- `configs/pipeline_config.json`
  - Set canonical template path and recalibrated normalized field boxes.
  - Tuned detection thresholds.
- `README.md`
  - Updated usage with canonical template behavior.

## Repo hygiene
- `.gitignore` remains valid ignore text.
- Text files (`.py/.json/.md/.txt/.yaml/.yml`) remain normal git files; no LFS pointer text introduced.
