# Repository Technical Audit Summary

## Initial state
- Repository was effectively empty (no existing OCR pipeline modules, configs, or tests).

## Reused components
- No reusable in-repo CV/OCR pipeline code existed to reuse.

## Replaced components
- None replaced; full stack implemented from scratch due to empty baseline.

## Key failure modes considered
- Scene image unreadable/empty.
- Card detection fails due to insufficient features.
- Homography fails or low-confidence quadrilateral.
- OCR backend not installed.
- Face not found in selfie or ID portrait.
- Missing template or missing input image directories.

## Final architecture chosen
- Python modular package (`egyptian_id_ocr`) with config-driven stages.
- Hybrid card detection: ORB+RANSAC homography plus contour quadrilateral fallback with rotation retries.
- Perspective warp + orientation selection.
- Template-geometry field extraction via configurable normalized boxes.
- Field-specific preprocessing and layered OCR backend fallback.
- Arabic normalization + digit normalization + Egyptian ID logical validation.
- Face verification wrapper with offline embedding backend fallback.
- Batch evaluator writing CSV/JSON/Markdown summaries.
