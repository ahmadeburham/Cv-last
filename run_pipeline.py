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
    p.add_argument("--selfie-dir", help="Batch selfie dir (ignored, kept for compatibility)")
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


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _failed_result(image: str, selfie: str | None, template: str, reason: str, role: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "input_image": image,
        "selfie_image": selfie or "",
        "template_image": template,
        "input_role": role,
        "id_present": False,
        "detection_confidence": 0.0,
        "selected_rotation": 0,
        "card_quad": [],
        "ocr": {
            "name": {"raw": "", "normalized": ""},
            "address": {"raw": "", "normalized": ""},
            "id_number": {"raw": "", "normalized_digits": "", "valid": False},
            "birth_date": {"raw": "", "normalized_digits": "", "valid": False},
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


def _classify_role(path: Path, template_path: Path, selfie_path: Path | None) -> str:
    rp = path.resolve()
    if rp == template_path.resolve():
        return "template"
    if selfie_path and rp == selfie_path.resolve():
        return "selfie"
    name = rp.name.lower()
    if any(t in name for t in ["template", "main_temp"]):
        return "template"
    if "selfie" in name:
        return "selfie"
    if any(t in name for t in ["ref", "reference", "sample_id"]):
        return "reference_card"
    return "scene_test"


def _write_eval(results: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_results = [r for r in results if r.get("input_role") == "scene_test"]
    csv_path = out_dir / "results_per_image.csv"
    csv_path.write_text(
        "input_image,status,id_present,detection_confidence,face_status,time_total,input_role\n"
        + "\n".join(
            f"{r.get('input_image','')},{r.get('status','')},{r.get('id_present',False)},{r.get('detection_confidence',0.0)},{r.get('face_match',{}).get('status','')},{r.get('timings',{}).get('total','')},{r.get('input_role','')}"
            for r in scene_results
        ),
        encoding="utf-8",
    )
    summary = {
        "num_images": len(scene_results),
        "excluded_non_scene_inputs": len(results) - len(scene_results),
        "detections_success": sum(1 for r in scene_results if r.get("id_present")),
        "alignments_success": sum(1 for r in scene_results if r.get("artifacts", {}).get("aligned_card")),
        "ocr_name_non_empty": sum(1 for r in scene_results if r.get("ocr", {}).get("name", {}).get("normalized")),
        "ocr_address_non_empty": sum(1 for r in scene_results if r.get("ocr", {}).get("address", {}).get("normalized")),
        "ocr_id_non_empty": sum(1 for r in scene_results if r.get("ocr", {}).get("id_number", {}).get("normalized_digits")),
        "ocr_birth_non_empty": sum(1 for r in scene_results if r.get("ocr", {}).get("birth_date", {}).get("normalized_digits")),
        "face_match_success_count": sum(1 for r in scene_results if r.get("face_match", {}).get("status") == "success"),
        "note": "Execution performed; full CV pipeline blocked when dependencies are unavailable.",
    }
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(
        "# Evaluation Summary\n"
        f"- Scene-test images processed: {summary['num_images']}\n"
        f"- Non-scene images excluded: {summary['excluded_non_scene_inputs']}\n"
        f"- Detection success: {summary['detections_success']}\n"
        f"- Alignment success: {summary['alignments_success']}\n"
        f"- Face match success: {summary['face_match_success_count']}\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)

    canonical_template = Path("main_temp.jpeg")
    if not canonical_template.exists():
        raise SystemExit("Canonical template missing: main_temp.jpeg")

    template = Path(args.template) if args.template else canonical_template
    if template.name != "main_temp.jpeg":
        raise SystemExit("Template must be main_temp.jpeg")
    output_dir = Path(args.output_dir or config.get("output_dir", "artifacts/runs"))

    try:
        from egyptian_id_ocr.config import PipelineConfig
        from egyptian_id_ocr.pipeline import EgyptianIDPipeline, run_batch, write_evaluation, classify_input_role

        cfg = PipelineConfig.from_json(args.config)
        if args.image:
            role = classify_input_role(Path(args.image), template, Path(args.selfie) if args.selfie else None)
            if role != "scene_test":
                raise SystemExit(f"Input role must be scene_test for single run; got {role}")
            run_dir = output_dir / Path(args.image).stem
            pipe = EgyptianIDPipeline(cfg)
            result = pipe.process(args.image, args.selfie, str(template), str(run_dir))
            result["input_role"] = role
            write_evaluation([result], "artifacts/eval")
            return

        if not args.input_dir:
            raise SystemExit("No input source. Use --image or --input-dir")
        results = run_batch(cfg, args.input_dir, args.selfie, str(template), str(output_dir))
        write_evaluation(results, "artifacts/eval")
    except Exception as exc:
        reason = f"dependency_or_runtime_failure:{exc}"
        results: list[dict[str, Any]] = []
        selfie_p = Path(args.selfie) if args.selfie else None

        if args.image:
            img_p = Path(args.image)
            role = _classify_role(img_p, template, selfie_p)
            run_dir = output_dir / img_p.stem
            run_dir.mkdir(parents=True, exist_ok=True)
            if img_p.exists():
                shutil.copy2(img_p, run_dir / img_p.name)
            r = _failed_result(str(img_p), args.selfie, str(template), reason, role)
            _write_json(run_dir / "result.json", r)
            results.append(r)
        else:
            input_dir = Path(args.input_dir) if args.input_dir else Path(".")
            for img in _discover_images(input_dir):
                role = _classify_role(img, template, selfie_p)
                if role != "scene_test":
                    continue
                run_dir = output_dir / img.stem
                run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, run_dir / img.name)
                r = _failed_result(str(img), args.selfie, str(template), reason, role)
                _write_json(run_dir / "result.json", r)
                results.append(r)
        _write_eval(results, Path("artifacts/eval"))


if __name__ == "__main__":
    main()
