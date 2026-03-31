# Final Change Log

## Added
- `egyptian_id_ocr/__init__.py`: package init.
- `egyptian_id_ocr/config.py`: dataclass-based config model and loader.
- `egyptian_id_ocr/utils.py`: image I/O, normalization, ID validation, JSON helpers.
- `egyptian_id_ocr/detection.py`: card detection (feature + contour), rotation handling, warping.
- `egyptian_id_ocr/ocr.py`: preprocessing variants and OCR fallback wrapper.
- `egyptian_id_ocr/face.py`: face detection and verification fallback stack.
- `egyptian_id_ocr/pipeline.py`: end-to-end pipeline, batch runner, evaluation writer.
- `configs/pipeline_config.json`: normalized field boxes and thresholds.
- `run_pipeline.py`: CLI for single and batch processing.
- `tests/test_utils.py`: normalization and ID validation tests.
- `tests/test_pipeline_components.py`: loading, pipeline result schema, CLI parse smoke.
- `requirements.txt`: runtime/test dependencies.
- `README.md`: architecture, install, run, evaluation docs.
- `artifacts/audit_summary.md`: repo audit and architecture rationale.

## Why
- End-to-end Egyptian ID-specific OCR + verification system was required and not present.
- Added robust local/offline processing path with explicit failure reporting and debug artifacts.
