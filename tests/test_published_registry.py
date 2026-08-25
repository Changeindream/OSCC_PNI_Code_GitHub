from pathlib import Path

from oscc_pni.app import load_inference_settings, load_registry
from oscc_pni.models.published_mil import PUBLISHED_ARCHITECTURES

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "model_registry.yaml"


def test_released_registry_is_complete_and_portable() -> None:
    registry = load_registry(REGISTRY)
    assert {settings["architecture"] for settings in registry.values()} == set(
        PUBLISHED_ARCHITECTURES
    )
    for settings in registry.values():
        checkpoint = Path(settings["checkpoint"])
        assert checkpoint.parent == ROOT / "weights"
        assert checkpoint.suffix == ".pt"


def test_inference_settings_match_deployed_interface() -> None:
    settings = load_inference_settings(REGISTRY)
    assert settings["class_names"] == ("Non-PNI", "PNI")
    assert settings["image_size"] == 224
    assert settings["max_slices_per_patient"] == 20
