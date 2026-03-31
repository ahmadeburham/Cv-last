from __future__ import annotations

import argparse
from pathlib import Path

from egyptian_id_ocr.config import PipelineConfig
from egyptian_id_ocr.pipeline import EgyptianIDPipeline, run_batch, write_evaluation, discover_images


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


def find_default_template() -> str:
    candidates = []
    for folder in [Path("ID template"), Path("id_template"), Path("template"), Path("templates")]:
        if folder.exists():
            candidates.extend(discover_images(folder))
    return str(candidates[0]) if candidates else ""


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig.from_json(args.config)
    template = args.template or cfg.template_path or find_default_template()
    if not template:
        raise SystemExit("No template provided/found. Use --template or config.template_path")
    output_dir = args.output_dir or cfg.output_dir

    if args.image:
        run_dir = Path(output_dir) / Path(args.image).stem
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

    results = run_batch(cfg, input_dir, args.selfie_dir, template, output_dir)
    write_evaluation(results, "artifacts/eval")


if __name__ == "__main__":
    main()
