# Repository Technical Audit Summary

## Reused
- Kept the existing hybrid classical-CV architecture (ORB homography + contour fallback) and modular package split.
- Kept config-driven normalized field extraction and JSON output contract.

## Replaced / strengthened
- Detection scoring now combines aspect, area ratio, rectangularity, border-edge support, and homography plausibility.
- Contour Canny thresholds now come from config (`canny_low`, `canny_high`) instead of hardcoded constants.
- Added post-warp sanity validation to reject implausible aligned cards.
- Replaced brightness-only orientation with a multi-signal orientation scorer (face presence, text density, horizontal text cues, layout consistency) and debug score dump.
- OCR winner selection is now field-aware (digits/date plausibility for numeric fields; Arabic/garbage/multi-word quality for text fields) and no longer confidence-only.
- EasyOCR reader is cached once per process.
- Face histogram fallback now returns `weak_fallback` with `verified: null`.
- Status semantics are now honest: `success` / `partial` / `failed` are derived from geometry + OCR + face quality.
- Added guarded final write path: partial results are always persisted even on downstream exceptions.
- Batch benchmarking excludes non-scene inputs via explicit role classification (`scene_test`, `template`, `selfie`, `reference_card`).

## Key blockers observed during rerun
1. Runtime dependency loading fails due missing `libGL.so.1` (OpenCV import dependency).
2. Provided test directory path had no discoverable images during batch rerun in this environment.
3. `main_temp.jpeg` must exist and is enforced by runner; missing file now fails explicitly.

## Final architecture chosen
- Preserve the hybrid CV pipeline and strengthen candidate ranking, warp plausibility checks, orientation selection, crop validation, and OCR candidate scoring.
- Preserve dependency-safe execution behavior to always emit auditable `result.json` files with explicit error reasons.
