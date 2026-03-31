# Egyptian Arabic ID OCR + Verification Pipeline

## Architecture
This repository implements a production-minded Egyptian national ID front-side pipeline using template-aware geometry and field-specific OCR:
1. Scene load/validation
2. Card detection with feature homography + contour quads + rotation retries
3. Perspective rectification to canonical template size
4. Orientation scoring and correction
5. Template-coordinate field extraction (portrait, birth date, name, address, ID number)
6. Field-specific preprocessing variants and OCR backend fallback
7. Arabic normalization + numeric normalization + Egyptian ID sanity checks
8. Face verification (face_recognition if available, histogram fallback)
9. Structured JSON output + exhaustive debug artifacts
10. Batch evaluation artifact generation

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

System packages often needed:
- `tesseract-ocr` and Arabic language data for pytesseract.

## Run (single)
```bash
python run_pipeline.py --image <scene.jpg> --selfie <selfie.jpg> --template <template.jpg> --output-dir artifacts/runs
```

## Run (batch)
```bash
python run_pipeline.py --input-dir <images_dir> --selfie-dir <selfie_dir> --template <template.jpg> --output-dir artifacts/runs
```

## Evaluation output
The pipeline writes:
- `artifacts/eval/results_per_image.csv`
- `artifacts/eval/summary.json`
- `artifacts/eval/summary.md`

## Output structure per image
- `original_image.*`
- `detection_overlay.jpg`
- `aligned_card.jpg`
- `oriented_card.jpg`
- `field_overlay.jpg`
- `fields/*.jpg`
- `preprocess/<field>/*.jpg`
- `id_face.jpg` + `selfie_face.jpg` (if available)
- `debug/ocr_dump.txt`, `debug/ocr_variants.json`
- `result.json`
- `summary.md`

## Tuning
- Edit `configs/pipeline_config.json` field boxes (normalized coordinates).
- Adjust OCR backend preference and detection thresholds in config.

## Known limitations
- No cloud OCR; purely local backends.
- Quality depends on local OCR packages and image quality.
- CER/WER requires explicit ground truth labels.
