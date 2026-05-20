import subprocess
import sys
from pathlib import Path

import pytest

from spatialclaw import cli_app


ROOT = Path(__file__).resolve().parent.parent


def invoke_cli_run(monkeypatch, argv: list[str], output_dir: Path):
    module = cli_app
    captured = {}

    def fake_run_skill(skill, **kwargs):
        captured["skill"] = skill
        captured.update(kwargs)
        return {
            "skill": skill,
            "success": True,
            "exit_code": 0,
            "output_dir": str(output_dir),
            "files": [],
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
        }

    monkeypatch.setattr(module, "run_skill", fake_run_skill)
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return captured


def test_main_forwards_spatial_omics_method_flags(monkeypatch, tmp_path):
    omics1 = tmp_path / "omics1.h5ad"
    omics2 = tmp_path / "omics2.h5ad"
    output_dir = tmp_path / "out"
    omics1.write_text("x")
    omics2.write_text("y")
    captured = invoke_cli_run(
        monkeypatch,
        [
            "spatialclaw.py",
            "run",
            "spatial-omics-integrate",
            "--input",
            str(omics1),
            "--omics2",
            str(omics2),
            "--output",
            str(output_dir),
            "--method",
            "spaddm",
            "--data-type",
            "Spatial-epigenome-transcriptome",
            "--omics1-type",
            "rna",
            "--omics2-type",
            "atac",
            "--clustering-method",
            "leiden",
            "--n-latent",
            "32",
            "--n-epochs",
            "5",
            "--random-seed",
            "123",
        ],
        output_dir,
    )

    assert captured["skill"] == "spatial-omics-integrate"
    assert captured["input_path"] == str(omics1)
    assert captured["output_dir"] == str(output_dir)
    assert captured["extra_args"] == [
        "--omics2",
        str(omics2),
        "--method",
        "spaddm",
        "--data-type",
        "Spatial-epigenome-transcriptome",
        "--omics1-type",
        "rna",
        "--omics2-type",
        "atac",
        "--clustering-method",
        "leiden",
        "--n-latent",
        "32",
        "--n-epochs",
        "5",
        "--random-seed",
        "123",
    ]


def test_main_forwards_spatial_modality_pearlst_flags(monkeypatch, tmp_path):
    sample_dir = tmp_path / "151673"
    output_dir = tmp_path / "out"
    ground_truth = tmp_path / "labels.tsv"
    simclr_features = tmp_path / "simclr.csv"
    simclr_model = tmp_path / "simclr_model.pth"
    sample_dir.mkdir()
    ground_truth.write_text("barcode\tlabel\n")
    simclr_features.write_text("barcode,1\n")
    simclr_model.write_text("weights")
    captured = invoke_cli_run(
        monkeypatch,
        [
            "spatialclaw.py",
            "run",
            "spatial-modality-integrate",
            "--input",
            str(sample_dir),
            "--output",
            str(output_dir),
            "--method",
            "pearlst",
            "--ground-truth",
            str(ground_truth),
            "--simclr-features",
            str(simclr_features),
            "--simclr-model",
            str(simclr_model),
        ],
        output_dir,
    )

    assert captured["skill"] == "spatial-modality-integrate"
    assert captured["input_path"] == str(sample_dir)
    assert captured["extra_args"] == [
        "--method",
        "pearlst",
        "--ground-truth",
        str(ground_truth),
        "--simclr-features",
        str(simclr_features),
        "--simclr-model",
        str(simclr_model),
    ]


def test_main_forwards_spatial_omics_device_flag(monkeypatch, tmp_path):
    omics1 = tmp_path / "omics1.h5ad"
    omics2 = tmp_path / "omics2.h5ad"
    output_dir = tmp_path / "out"
    omics1.write_text("x")
    omics2.write_text("y")
    captured = invoke_cli_run(
        monkeypatch,
        [
            "spatialclaw.py",
            "run",
            "spatial-omics-integrate",
            "--input",
            str(omics1),
            "--omics2",
            str(omics2),
            "--output",
            str(output_dir),
            "--device",
            "gpu",
        ],
        output_dir,
    )

    assert captured["extra_args"] == [
        "--omics2",
        str(omics2),
        "--device",
        "gpu",
    ]


@pytest.mark.parametrize(
    ("skill", "input_name", "extra_tokens", "expected_extra_args"),
    [
        (
            "spatial-preprocessing",
            "sample.h5ad",
            ["--species", "human", "--min-genes", "150", "--n-pcs", "30"],
            ["--species", "human", "--min-genes", "150", "--n-pcs", "30"],
        ),
        (
            "spatial-domain-identification",
            "sample.h5ad",
            ["--method", "spagcn", "--n-domains", "7", "--refine"],
            ["--method", "spagcn", "--n-domains", "7", "--refine"],
        ),
        (
            "spatial-cell-annotation",
            "sample.h5ad",
            ["--reference", "reference.h5ad", "--cell-type-key", "cell_type", "--model", "model.pt"],
            ["--reference", "reference.h5ad", "--cell-type-key", "cell_type", "--model", "model.pt"],
        ),
        (
            "spatial-enrichment",
            "sample.h5ad",
            ["--analysis-type", "go", "--go-ontology", "BP", "--groupby", "leiden", "--organism", "human", "--no-save-h5ad"],
            ["--analysis-type", "go", "--go-ontology", "BP", "--groupby", "leiden", "--organism", "human", "--no-save-h5ad"],
        ),
        (
            "spatial-cnv",
            "sample.h5ad",
            ["--reference-key", "cell_type", "--reference-cat", "Normal", "Control", "--window-size", "100", "--step", "10"],
            ["--reference-key", "cell_type", "--reference-cat", "Normal", "Control", "--window-size", "100", "--step", "10"],
        ),
    ],
)
def test_main_forwards_non_integrate_skill_specific_flags_after_global_parser_slimming(
    monkeypatch,
    tmp_path,
    skill,
    input_name,
    extra_tokens,
    expected_extra_args,
):
    input_path = tmp_path / input_name
    output_dir = tmp_path / f"{skill}-out"
    input_path.write_text("x")

    for token in extra_tokens:
        if token.startswith("--"):
            continue
        if token in {"human", "spagcn", "go", "BP", "leiden", "cell_type", "Normal", "Control", "100", "10", "parametric", "ensembl", "symbol"}:
            continue
        candidate = tmp_path / token
        candidate.write_text("x")

    captured = invoke_cli_run(
        monkeypatch,
        [
            "spatialclaw.py",
            "run",
            skill,
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            *extra_tokens,
        ],
        output_dir,
    )

    assert captured["skill"] == skill
    assert captured["input_path"] == str(input_path)
    assert captured["extra_args"] == expected_extra_args


def test_run_skill_preflights_required_multiomics_inputs(monkeypatch, tmp_path):
    module = cli_app

    def should_not_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when --omics2 is missing")

    monkeypatch.setattr(module.subprocess, "run", should_not_run)

    result = module.run_skill(
        "spatial-omics-integrate",
        input_path=str(tmp_path / "omics1.h5ad"),
        output_dir=str(tmp_path / "out"),
    )

    assert result["success"] is False
    assert "--omics2" in result["stderr"]


def test_run_skill_requires_real_alternative_input_flags(monkeypatch, tmp_path):
    module = cli_app

    def should_not_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called without --input-list")

    monkeypatch.setattr(module.subprocess, "run", should_not_run)

    result = module.run_skill(
        "spatial-modality-integrate",
        output_dir=str(tmp_path / "out"),
        extra_args=["--mode", "integration"],
    )

    assert result["success"] is False
    assert "--input-list" in result["stderr"]


def test_run_skill_accepts_registry_alternative_input_mode(monkeypatch, tmp_path):
    module = cli_app
    input_list = tmp_path / "samples.txt"
    output_dir = tmp_path / "out"
    input_list.write_text("data/DLPFC/151673\n")
    captured = {}

    def fake_run(cmd, capture_output, text, cwd, env):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_skill(
        "spatial-modality-integrate",
        output_dir=str(output_dir),
        extra_args=["--mode", "integration", "--input-list", str(input_list)],
    )

    assert result["success"] is True
    assert "--input" not in captured["cmd"]
    assert "--input-list" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--input-list") + 1] == str(input_list.resolve())


def test_run_skill_resolves_required_extra_input_paths(monkeypatch, tmp_path):
    module = cli_app
    omics1 = tmp_path / "omics1.h5ad"
    omics2 = tmp_path / "omics2.h5ad"
    output_dir = tmp_path / "out"
    omics1.write_text("x")
    omics2.write_text("y")
    captured = {}

    def fake_run(cmd, capture_output, text, cwd, env):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_skill(
        "spatial-omics-integrate",
        input_path=str(omics1),
        output_dir=str(output_dir),
        extra_args=["--method", "spatialglue", "--omics2", str(omics2)],
    )

    assert result["success"] is True
    assert captured["cmd"][captured["cmd"].index("--omics2") + 1] == str(omics2.resolve())


def test_run_skill_resolves_general_path_extra_flags(monkeypatch, tmp_path):
    module = cli_app
    input_path = tmp_path / "sample.h5ad"
    reference = tmp_path / "reference.h5ad"
    model = tmp_path / "model.pt"
    output_dir = tmp_path / "out"
    input_path.write_text("x")
    reference.write_text("ref")
    model.write_text("weights")
    captured = {}

    def fake_run(cmd, capture_output, text, cwd, env):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_skill(
        "spatial-cell-annotation",
        input_path=str(input_path),
        output_dir=str(output_dir),
        extra_args=["--reference", str(reference), "--cell-type-key", "cell_type", "--model", str(model)],
    )

    assert result["success"] is True
    assert captured["cmd"][captured["cmd"].index("--reference") + 1] == str(reference.resolve())
    assert captured["cmd"][captured["cmd"].index("--model") + 1] == str(model.resolve())


def test_run_skill_keeps_multi_value_passthrough_flags(monkeypatch, tmp_path):
    module = cli_app
    input_path = tmp_path / "sample.h5ad"
    output_dir = tmp_path / "out"
    input_path.write_text("x")
    captured = {}

    def fake_run(cmd, capture_output, text, cwd, env):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_skill(
        "spatial-cnv",
        input_path=str(input_path),
        output_dir=str(output_dir),
        extra_args=["--reference-key", "cell_type", "--reference-cat", "Normal", "Control"],
    )

    assert result["success"] is True
    ref_idx = captured["cmd"].index("--reference-cat")
    assert captured["cmd"][ref_idx + 1:ref_idx + 3] == ["Normal", "Control"]


def test_run_skill_rejects_unsupported_passthrough_flags(tmp_path):
    module = cli_app

    result = module.run_skill(
        "spatial-omics-integrate",
        input_path=str(tmp_path / "omics1.h5ad"),
        output_dir=str(tmp_path / "out"),
        extra_args=["--not-a-real-flag", "value"],
    )

    assert result["success"] is False
    assert "--not-a-real-flag" in result["stderr"]
