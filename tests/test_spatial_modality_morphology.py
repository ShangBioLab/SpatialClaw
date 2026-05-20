from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "spatial"
    / "spatial-modality-integrate"
    / "spatial_modality_integrate.py"
)


@pytest.fixture(scope="module")
def modality_module():
    pytest.importorskip("scanpy")
    pytest.importorskip("deepstkit")

    spec = spec_from_file_location("spatial_modality_integrate", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_info(name: str, has_image: bool) -> dict:
    return {
        "sample_name": name,
        "image_path": Path(f"/tmp/{name}.png") if has_image else None,
    }


def test_single_sample_auto_enables_morphology_when_image_exists(modality_module):
    resolved = modality_module._resolve_morphology_setting(
        requested=None,
        sample_infos=[_sample_info("sample_a", True)],
        mode="single",
    )

    assert resolved["enabled"] is True
    assert resolved["status_label"] == "Enabled (auto-detected from tissue image)"


def test_single_sample_auto_disables_morphology_without_image(modality_module):
    resolved = modality_module._resolve_morphology_setting(
        requested=None,
        sample_infos=[_sample_info("sample_a", False)],
        mode="single",
    )

    assert resolved["enabled"] is False
    assert resolved["status_label"] == "Disabled (no tissue image available)"


def test_integration_auto_requires_images_for_all_samples(modality_module):
    resolved = modality_module._resolve_morphology_setting(
        requested=None,
        sample_infos=[
            _sample_info("sample_a", True),
            _sample_info("sample_b", False),
        ],
        mode="integration",
    )

    assert resolved["enabled"] is False
    assert resolved["status_label"] == "Disabled (not all samples have tissue images)"
