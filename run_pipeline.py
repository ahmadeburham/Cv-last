from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Egyptian ID OCR pipeline")
    p.add_argument("--config", default="configs/pipeline_config.json")
    p.add_argument("--image", help="Single scene image path")
    p.add_argument("--selfie", help="Single selfie image path")
    p.add_argument("--input-dir", help="Batch input image dir")
    p.add_argument("--selfie-dir", help="Batch selfie dir")
    p.add_argument("--template", required=False, help="Template image path")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def _load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _discover_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def find_default_template() -> str:
    candidates = []
    for folder in [Path("ID template"), Path("id_template"), Path("template"), Path("templates")]:
        candidates.extend(_discover_images(folder))
    return str(candidates[0]) if candidates else ""


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _failed_result(image: str, selfie: str | None, template: str, reason: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "input_image": image,
        "selfie_image": selfie or "",
        "template_image": template,
        "id_present": False,
        "detection_confidence": 0.0,
        "selected_rotation": 0,
        "card_quad": [],
        "ocr": {
            "name": {"raw": "", "normalized": ""},
            "address": {"raw": "", "normalized": ""},
            "id_number": {"raw": "", "normalized_digits": "", "valid": False},
            "birth_date": {"raw": "", "normalized_digits": "", "valid": False}
        },
        "face_match": {
            "verified": None,
            "score": None,
            "threshold": 0.55,
            "status": "failed",
            "reason": reason,
        },
        "artifacts": {},
        "errors": [reason],
        "warnings": [],
        "timings": {},
    }


def _write_eval(results: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results_per_image.csv"
    csv_path.write_text("input_image,status,id_present,detection_confidence,face_status,time_total\n" + "\n".join(
        f"{r.get('input_image','')},{r.get('status','')},{r.get('id_present',False)},{r.get('detection_confidence',0.0)},{r.get('face_match',{}).get('status','')},{r.get('timings',{}).get('total','')}"
        for r in results
    ), encoding="utf-8")
    summary = {
        "num_images": len(results),
        "detections_success": sum(1 for r in results if r.get("id_present")),
        "alignments_success": sum(1 for r in results if r.get("artifacts", {}).get("aligned_card")),
        "ocr_name_non_empty": sum(1 for r in results if r.get("ocr", {}).get("name", {}).get("normalized")),
        "ocr_address_non_empty": sum(1 for r in results if r.get("ocr", {}).get("address", {}).get("normalized")),
        "ocr_id_non_empty": sum(1 for r in results if r.get("ocr", {}).get("id_number", {}).get("normalized_digits")),
        "ocr_birth_non_empty": sum(1 for r in results if r.get("ocr", {}).get("birth_date", {}).get("normalized_digits")),
        "face_match_success_count": sum(1 for r in results if r.get("face_match", {}).get("status") == "success"),
        "note": "Execution performed; full CV pipeline blocked when dependencies are unavailable."
    }
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(
        "# Evaluation Summary\n"
        f"- Images processed: {summary['num_images']}\n"
        f"- Detection success: {summary['detections_success']}\n"
        f"- Alignment success: {summary['alignments_success']}\n"
        f"- Face match success: {summary['face_match_success_count']}\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    template = args.template or config.get("template_path") or find_default_template()
    if not template:
        raise SystemExit("No template provided/found. Use --template or config.template_path")
    output_dir = Path(args.output_dir or config.get("output_dir", "artifacts/runs"))

    try:
        from egyptian_id_ocr.config import PipelineConfig
        from egyptian_id_ocr.pipeline import EgyptianIDPipeline, run_batch, write_evaluation

        cfg = PipelineConfig.from_json(args.config)
        if args.image:
            run_dir = output_dir / Path(args.image).stem
            pipe = EgyptianIDPipeline(cfg)
            result = pipe.process(args.image, args.selfie, template, str(run_dir))
            write_evaluation([result], "artifacts/eval")
            return

        input_dir = args.input_dir
        if not input_dir:
            for candidate in ["output", "test", "tests/data", "images"]:
                if Path(candidate).exists():
                    input_dir = candidate
                    break
        if not input_dir:
            raise SystemExit("No input source. Use --image or --input-dir")
        results = run_batch(cfg, input_dir, args.selfie_dir, template, str(output_dir))
        write_evaluation(results, "artifacts/eval")
    except Exception as exc:
        reason = f"dependency_or_runtime_failure:{exc}"
        results: list[dict[str, Any]] = []
        if args.image:
            run_dir = output_dir / Path(args.image).stem
            run_dir.mkdir(parents=True, exist_ok=True)
            if Path(args.image).exists():
                shutil.copy2(args.image, run_dir / Path(args.image).name)
            r = _failed_result(args.image, args.selfie, template, reason)
            _write_json(run_dir / "result.json", r)
            results.append(r)
        else:
            input_dir = args.input_dir
            if not input_dir:
                for candidate in ["output", "test", "tests/data", "images", "/tmp/user_uploaded_attachments"]:
                    if Path(candidate).exists():
                        input_dir = candidate
                        break
            if not input_dir:
                raise SystemExit(reason)
            for img in _discover_images(Path(input_dir)):
                run_dir = output_dir / img.stem
                run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, run_dir / img.name)
                r = _failed_result(str(img), None, template, reason)
                _write_json(run_dir / "result.json", r)
                results.append(r)
        _write_eval(results, Path("artifacts/eval"))


if __name__ == "__main__":
    main()
