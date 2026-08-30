import json
from pathlib import Path

from oscc_pni.app import load_inference_settings, load_registry
from oscc_pni.models.published_mil import PUBLISHED_ARCHITECTURES

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "model_registry.yaml"
WEIGHT_MANIFEST = ROOT / "weights" / "manifest.json"


def test_released_registry_is_complete_and_portable() -> None:
    registry = load_registry(REGISTRY)
    assert {settings["architecture"] for settings in registry.values()} == set(
        PUBLISHED_ARCHITECTURES
    )
    for settings in registry.values():
        checkpoint = Path(settings["checkpoint"])
        assert checkpoint.parent == ROOT / "weights"
        assert checkpoint.suffix == ".pt"
    assert {Path(settings["checkpoint"]).name for settings in registry.values()} == {
        "resnet_mil.pt",
        "densenet_mil.pt",
        "swin_mil.pt",
        "vit_mil.pt",
    }


def test_inference_settings_match_deployed_interface() -> None:
    settings = load_inference_settings(REGISTRY)
    assert settings["class_names"] == ("Non-PNI", "PNI")
    assert settings["image_size"] == 224
    assert settings["max_slices_per_patient"] == 20


def test_external_weight_manifest_matches_registry() -> None:
    registry = load_registry(REGISTRY)
    manifest = json.loads(WEIGHT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["storage"] == "huggingface"
    assert manifest["repository_id"] == "changeindream/oscc-pni-mil-checkpoints"
    assert len(manifest["revision"]) == 40
    records = {record["architecture"]: record for record in manifest["weights"]}
    assert set(records) == set(PUBLISHED_ARCHITECTURES)
    for settings in registry.values():
        record = records[settings["architecture"]]
        assert Path(settings["checkpoint"]).name == record["file"]
        assert int(record["bytes"]) > 0
        assert len(record["sha256"]) == 64
