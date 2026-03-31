import numpy as np
import cv2
from pathlib import Path

from egyptian_id_ocr.config import PipelineConfig
from egyptian_id_ocr.pipeline import EgyptianIDPipeline


def test_image_loading(tmp_path: Path):
    p = tmp_path / "x.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(p), img)
    from egyptian_id_ocr.utils import load_image

    out = load_image(p)
    assert out.shape == img.shape


def test_field_crop_and_json_schema(tmp_path: Path):
    scene = np.full((640, 1000, 3), 255, dtype=np.uint8)
    template = scene.copy()
    selfie = scene.copy()

    scene_p = tmp_path / "scene.jpg"
    template_p = tmp_path / "template.jpg"
    selfie_p = tmp_path / "selfie.jpg"
    cv2.imwrite(str(scene_p), scene)
    cv2.imwrite(str(template_p), template)
    cv2.imwrite(str(selfie_p), selfie)

    cfg = PipelineConfig.from_json("configs/pipeline_config.json")
    pipe = EgyptianIDPipeline(cfg)
    res = pipe.process(str(scene_p), str(selfie_p), str(template_p), str(tmp_path / "run"))
    assert "status" in res
    assert "ocr" in res
    assert "face_match" in res
    assert (tmp_path / "run" / "result.json").exists()


def test_cli_argument_parsing(monkeypatch):
    import run_pipeline

    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--image", "a.jpg", "--template", "t.jpg"])
    args = run_pipeline.parse_args()
    assert args.image == "a.jpg"
    assert args.template == "t.jpg"
