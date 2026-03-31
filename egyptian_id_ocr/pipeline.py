from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import csv
import json
import shutil
import cv2
import numpy as np

from .config import PipelineConfig, from_dict
from .detection import detect_card, rotate_image, warp_from_quad
from .ocr import preprocess_variants, run_ocr_with_fallback
from .face import compare_faces, detect_face_crop
from .utils import ensure_dir, load_image, normalize_arabic_text, only_digits, save_json, validate_egyptian_id, timed


class EgyptianIDPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    @staticmethod
    def from_config_path(path: str) -> "EgyptianIDPipeline":
        return EgyptianIDPipeline(PipelineConfig.from_json(path))

    def _default_result(self, image: str, selfie: str | None, template: str) -> dict:
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
                "birth_date": {"raw": "", "normalized_digits": "", "valid": False},
            },
            "face_match": {
                "verified": None,
                "score": None,
                "threshold": self.cfg.face.match_threshold,
                "status": "not_run",
                "reason": "",
            },
            "artifacts": {},
            "errors": [],
            "warnings": [],
            "timings": {},
        }

    def _write_text_dump(self, path: Path, result: dict) -> None:
        lines = [
            f"name_raw={result['ocr']['name']['raw']}",
            f"address_raw={result['ocr']['address']['raw']}",
            f"id_number_raw={result['ocr']['id_number']['raw']}",
            f"birth_date_raw={result['ocr']['birth_date']['raw']}",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def process(self, image_path: str, selfie_path: str | None, template_path: str, out_dir: str) -> dict:
        total_start = perf_counter()
        out = ensure_dir(out_dir)
        debug = ensure_dir(out / "debug")
        fields_dir = ensure_dir(out / "fields")
        prep_dir = ensure_dir(out / "preprocess")

        result = self._default_result(image_path, selfie_path, template_path)

        try:
            t0 = perf_counter()
            scene = load_image(image_path)
            template = load_image(template_path)
            original_copy = out / f"original_image{Path(image_path).suffix}"
            shutil.copy2(image_path, original_copy)
            result["artifacts"]["original_image"] = str(original_copy)
            result["timings"]["load"] = timed(t0, perf_counter())
        except Exception as exc:
            result["errors"].append(f"load_failed:{exc}")
            save_json(out / "result.json", result)
            return result

        det_start = perf_counter()
        det = detect_card(scene, template, self.cfg.detection.rotation_candidates, self.cfg.detection.min_card_area_ratio)
        result["timings"]["detect"] = timed(det_start, perf_counter())
        result["detection_confidence"] = float(det.confidence)
        result["selected_rotation"] = int(det.rotation)

        if not det.found or det.quad is None or det.confidence < self.cfg.detection.min_confidence:
            result["errors"].append(f"detection_failed:{det.reason}")
            result["status"] = "failed"
            save_json(out / "result.json", result)
            return result

        rotated = rotate_image(scene, det.rotation)
        overlay = rotated.copy()
        quad_int = det.quad.astype(int)
        cv2.polylines(overlay, [quad_int], True, (0, 255, 0), 3)
        overlay_path = out / "detection_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        result["artifacts"]["overlay"] = str(overlay_path)
        result["card_quad"] = det.quad.round(2).tolist()
        result["id_present"] = True

        align_start = perf_counter()
        aligned = warp_from_quad(rotated, det.quad, self.cfg.aligned_size)
        aligned_path = out / "aligned_card.jpg"
        cv2.imwrite(str(aligned_path), aligned)
        result["artifacts"]["aligned_card"] = str(aligned_path)
        result["timings"]["align"] = timed(align_start, perf_counter())

        orient_start = perf_counter()
        candidates = [(0, aligned), (90, rotate_image(aligned, 90)), (180, rotate_image(aligned, 180)), (270, rotate_image(aligned, 270))]
        scores: list[tuple[float, int, np.ndarray]] = []
        for ang, img in candidates:
            h, w = img.shape[:2]
            left = cv2.cvtColor(img[:, : w // 3], cv2.COLOR_BGR2GRAY).mean()
            right = cv2.cvtColor(img[:, w // 3 :], cv2.COLOR_BGR2GRAY).mean()
            score = float((right - left) / 255.0)
            scores.append((score, ang, img))
        _, best_ang, oriented = sorted(scores, key=lambda x: x[0], reverse=True)[0]
        oriented_path = out / "oriented_card.jpg"
        cv2.imwrite(str(oriented_path), oriented)
        result["artifacts"]["oriented_card"] = str(oriented_path)
        result["timings"]["orientation"] = timed(orient_start, perf_counter())

        fields_overlay = oriented.copy()
        fh, fw = oriented.shape[:2]
        crops: dict[str, np.ndarray] = {}
        for name, box in self.cfg.fields.items():
            x1 = int(box.x * fw)
            y1 = int(box.y * fh)
            x2 = int((box.x + box.w) * fw)
            y2 = int((box.y + box.h) * fh)
            crop = oriented[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
            crops[name] = crop
            cv2.rectangle(fields_overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cpath = fields_dir / f"{name}.jpg"
            cv2.imwrite(str(cpath), crop)
            result["artifacts"][f"{name}_crop"] = str(cpath)
        foverlay_path = out / "field_overlay.jpg"
        cv2.imwrite(str(foverlay_path), fields_overlay)
        result["artifacts"]["field_overlay"] = str(foverlay_path)

        ocr_start = perf_counter()
        ocr_dump = {}
        for fname in ["name", "address", "id_number", "birth_date"]:
            variants = preprocess_variants(fname, crops[fname], self.cfg.preprocess.scale_factor)
            fprep = ensure_dir(prep_dir / fname)
            for vn, vimg in variants.items():
                cv2.imwrite(str(fprep / f"{vn}.jpg"), vimg)
            ocr = run_ocr_with_fallback(variants, self.cfg.ocr.primary_backend, self.cfg.ocr.fallback_backends, self.cfg.ocr.tesseract_lang)
            best_img = variants.get(ocr.variant)
            if best_img is not None:
                cv2.imwrite(str(fprep / "best.jpg"), best_img)
            ocr_dump[fname] = asdict(ocr)

        result["ocr"]["name"]["raw"] = ocr_dump["name"]["text"]
        result["ocr"]["address"]["raw"] = ocr_dump["address"]["text"]
        result["ocr"]["id_number"]["raw"] = ocr_dump["id_number"]["text"]
        result["ocr"]["birth_date"]["raw"] = ocr_dump["birth_date"]["text"]

        result["ocr"]["name"]["normalized"] = normalize_arabic_text(result["ocr"]["name"]["raw"])
        result["ocr"]["address"]["normalized"] = normalize_arabic_text(result["ocr"]["address"]["raw"])

        id_digits = only_digits(result["ocr"]["id_number"]["raw"])
        result["ocr"]["id_number"]["normalized_digits"] = id_digits
        id_ok, _ = validate_egyptian_id(id_digits)
        result["ocr"]["id_number"]["valid"] = bool(id_ok)

        b_digits = only_digits(result["ocr"]["birth_date"]["raw"])
        result["ocr"]["birth_date"]["normalized_digits"] = b_digits
        result["ocr"]["birth_date"]["valid"] = len(b_digits) in {6, 8}

        self._write_text_dump(debug / "ocr_dump.txt", result)
        Path(debug / "ocr_variants.json").write_text(json.dumps(ocr_dump, ensure_ascii=False, indent=2), encoding="utf-8")
        result["timings"]["ocr"] = timed(ocr_start, perf_counter())

        face_start = perf_counter()
        if selfie_path:
            try:
                selfie = load_image(selfie_path)
                id_face, _ = detect_face_crop(crops["portrait"], self.cfg.face.min_face_size)
                sf_face, _ = detect_face_crop(selfie, self.cfg.face.min_face_size)
                if id_face is None or sf_face is None:
                    result["face_match"] = {
                        "verified": None,
                        "score": None,
                        "threshold": self.cfg.face.match_threshold,
                        "status": "failed",
                        "reason": "face_not_found",
                    }
                else:
                    idf = out / "id_face.jpg"
                    sff = out / "selfie_face.jpg"
                    cv2.imwrite(str(idf), id_face)
                    cv2.imwrite(str(sff), sf_face)
                    result["artifacts"]["id_face"] = str(idf)
                    result["artifacts"]["selfie_face"] = str(sff)
                    m = compare_faces(id_face, sf_face, self.cfg.face.match_threshold)
                    result["face_match"] = {
                        "verified": m.verified,
                        "score": m.score,
                        "threshold": m.threshold,
                        "status": m.status,
                        "reason": m.reason,
                    }
            except Exception as exc:
                result["face_match"] = {
                    "verified": None,
                    "score": None,
                    "threshold": self.cfg.face.match_threshold,
                    "status": "failed",
                    "reason": f"face_stage_error:{exc}",
                }
        else:
            result["warnings"].append("selfie_not_provided")

        result["timings"]["face"] = timed(face_start, perf_counter())
        result["timings"]["total"] = timed(total_start, perf_counter())
        result["status"] = "success"
        save_json(out / "result.json", result)
        (out / "summary.md").write_text(
            f"# Run Summary\n\n- status: {result['status']}\n- detection_confidence: {result['detection_confidence']}\n",
            encoding="utf-8",
        )
        return result


def discover_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()])


def run_batch(config: PipelineConfig, input_dir: str, selfie_dir: str | None, template_path: str, output_dir: str) -> list[dict]:
    pipe = EgyptianIDPipeline(config)
    inputs = discover_images(Path(input_dir))
    selfies = discover_images(Path(selfie_dir)) if selfie_dir else []
    selfie_map = {p.stem.lower(): p for p in selfies}
    out_root = ensure_dir(output_dir)
    results: list[dict] = []

    for img in inputs:
        selfie = selfie_map.get(img.stem.lower())
        run_dir = out_root / img.stem
        res = pipe.process(str(img), str(selfie) if selfie else None, template_path, str(run_dir))
        results.append(res)
    return results


def write_evaluation(results: list[dict], out_dir: str) -> None:
    out = ensure_dir(out_dir)
    csv_path = out / "results_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["input_image", "status", "id_present", "detection_confidence", "face_status", "time_total"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "input_image": r.get("input_image"),
                    "status": r.get("status"),
                    "id_present": r.get("id_present"),
                    "detection_confidence": r.get("detection_confidence"),
                    "face_status": r.get("face_match", {}).get("status"),
                    "time_total": r.get("timings", {}).get("total"),
                }
            )
    summary = {
        "num_images": len(results),
        "detections_success": sum(1 for r in results if r.get("id_present")),
        "alignments_success": sum(1 for r in results if r.get("artifacts", {}).get("aligned_card")),
        "ocr_name_non_empty": sum(1 for r in results if r.get("ocr", {}).get("name", {}).get("normalized")),
        "ocr_address_non_empty": sum(1 for r in results if r.get("ocr", {}).get("address", {}).get("normalized")),
        "ocr_id_non_empty": sum(1 for r in results if r.get("ocr", {}).get("id_number", {}).get("normalized_digits")),
        "ocr_birth_non_empty": sum(1 for r in results if r.get("ocr", {}).get("birth_date", {}).get("normalized_digits")),
        "face_match_success_count": sum(1 for r in results if r.get("face_match", {}).get("status") == "success"),
        "note": "No CER/WER computed unless labels exist.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(
        "\n".join(
            [
                "# Evaluation Summary",
                f"- Images processed: {summary['num_images']}",
                f"- Detection success: {summary['detections_success']}",
                f"- Alignment success: {summary['alignments_success']}",
                f"- Face match success: {summary['face_match_success_count']}",
                "- CER/WER/field accuracy not computed: no ground-truth labels discovered.",
            ]
        ),
        encoding="utf-8",
    )
