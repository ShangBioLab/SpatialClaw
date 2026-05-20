import ast
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from spatialclaw.core.registry import (
    API_ONLY_SPATIAL_SKILLS,
    build_skill_llm_cli_args,
    normalize_skill_llm_args,
    registry,
)

_STANDARD_CLI_FLAGS = {"--input", "--output", "--demo"}


def _argparse_cli_flags(script_path: Path) -> set[str]:
    return set(_argparse_argument_contracts(script_path))


def _literal_name_bindings(tree: ast.AST) -> dict[str, object]:
    bindings: dict[str, object] = {}
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            value = _literal_call_value(node.value, bindings)
        if value == "DYNAMIC":
            continue
        bindings[target.id] = value
    return bindings


def _literal_call_value(node: ast.AST, bindings: dict[str, object]) -> object:
    if isinstance(node, ast.Name) and node.id in bindings:
        return bindings[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"list", "tuple", "set"}:
        if len(node.args) == 1:
            value = _literal_call_value(node.args[0], bindings)
            if value == "DYNAMIC":
                return "DYNAMIC"
            if isinstance(value, dict):
                value = value.keys()
            return list(value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "keys":
        owner = node.func.value
        if isinstance(owner, ast.Name) and isinstance(bindings.get(owner.id), dict):
            return list(bindings[owner.id].keys())
    try:
        return ast.literal_eval(node)
    except Exception:
        return "DYNAMIC"


def _argparse_argument_contracts(script_path: Path) -> dict[str, dict[str, object]]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    bindings = _literal_name_bindings(tree)
    flags: dict[str, dict[str, object]] = {}

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue

        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                flags.setdefault(
                    arg.value,
                    {
                        "choices": None,
                        "default": None,
                        "required": False,
                        "action": None,
                    },
                )

        names = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        ]
        if not names:
            continue
        contract = flags[names[0]]
        for keyword in node.keywords:
            if keyword.arg in contract:
                contract[keyword.arg] = _literal_call_value(keyword.value, bindings)

    return flags


def _markdown_cli_flags(skill_md: Path) -> set[str]:
    return set(
        re.findall(
            r"`(--[A-Za-z0-9][A-Za-z0-9-]*)`",
            skill_md.read_text(encoding="utf-8"),
        )
    )


def _markdown_text(skill_md: Path) -> str:
    return skill_md.read_text(encoding="utf-8")


def test_registry_loaded():
    registry.load_all()
    assert "spatial-preprocessing" in registry.skills
    assert "spatial" in registry.domains
    assert len(registry.skills) > 0
    assert len(registry.domains) > 0
    assert registry.domains["spatial"]["skill_count"] == len(registry.skills)


def test_api_only_spatial_skills_are_not_cli_registered():
    registry.load_all()

    assert API_ONLY_SPATIAL_SKILLS.isdisjoint(registry.skills)


def test_spatial_modality_integrate_metadata():
    registry.load_all()
    skill = registry.skills["spatial-modality-integrate"]
    method_schema = skill["llm_argument_schema"]["method"]
    platform_schema = skill["llm_argument_schema"]["platform"]

    assert skill["input_mode"] == "directory"
    assert skill["requires_input"] is True
    assert skill["allows_alternative_inputs"] is True
    assert skill["alternative_input_flags"] == {"--input-list"}
    assert "--input" not in skill["allowed_extra_flags"]
    assert "--method" in skill["allowed_extra_flags"]
    assert "--input-list" in skill["allowed_extra_flags"]
    assert "--no-morphological" in skill["allowed_extra_flags"]
    assert "--ground-truth" in skill["allowed_extra_flags"]
    assert "--simclr-features" in skill["allowed_extra_flags"]
    assert "--simclr-model" in skill["allowed_extra_flags"]
    assert skill["path_extra_flags"] == {
        "--ground-truth",
        "--simclr-features",
        "--simclr-model",
    }
    assert "demo_args" not in skill
    assert skill["input_examples"]["primary"] == "/data/DLPFC/151673"
    assert method_schema["allowed_values"] == ("deepst", "pearlst")
    assert method_schema["value_aliases"]["pearist"] == "pearlst"
    assert platform_schema["allowed_values"] == (
        "Visium",
        "Stereo-seq",
        "Slide-seq",
        "Slide-seqV2",
        "MERFISH",
        "STARmap",
    )


def test_spatial_omics_integrate_metadata():
    registry.load_all()
    skill = registry.skills["spatial-omics-integrate"]
    method_schema = skill["llm_argument_schema"]["method"]
    platform_schema = skill["llm_argument_schema"]["platform"]

    assert skill["input_mode"] == "file"
    assert skill["requires_input"] is True
    assert skill["demo_args"] == ["--demo"]
    assert skill["required_extra_inputs"] == [
        {
            "flag": "--omics2",
            "kind": "file",
            "label": "secondary spatial modality file",
        }
    ]
    assert "--omics2" in skill["allowed_extra_flags"]
    assert "--method" in skill["allowed_extra_flags"]
    assert "--clustering-method" in skill["allowed_extra_flags"]
    assert "--n-latent" in skill["allowed_extra_flags"]
    assert skill["input_examples"]["usage"] == "--input rna.h5ad --omics2 protein.h5ad --method spatialglue"
    assert method_schema["allowed_values"] == ("spatialglue", "spaddm")
    assert method_schema["value_aliases"]["spa-ddm"] == "spaddm"
    assert platform_schema["allowed_values"] == (
        "auto",
        "10x",
        "Stereo-CITE-seq",
        "SPOTS",
        "spatial-epigenome",
        "Spatial-epigenome-transcriptome",
        "Visium CytAssist",
    )
    assert platform_schema["value_aliases"]["spatial"] == "auto"
    assert platform_schema["value_aliases"]["visium"] == "auto"
    assert platform_schema["drop_invalid"] is True


def test_normalize_skill_llm_args_maps_structured_argument_names():
    normalized = normalize_skill_llm_args(
        "spatial-omics-integrate",
        {
            "save_dir": "results/demo-out",
            "data_type": "spatial",
        },
    )

    assert normalized["output_dir"] == "results/demo-out"
    assert normalized["platform"] == "auto"
    assert "save_dir" not in normalized
    assert "data_type" not in normalized


def test_normalize_skill_llm_args_drops_invalid_platform():
    normalized = normalize_skill_llm_args(
        "spatial-omics-integrate",
        {"platform": "not-a-real-platform"},
    )

    assert "platform" not in normalized


def test_normalize_skill_llm_args_drops_non_positive_n_epochs():
    normalized = normalize_skill_llm_args(
        "spatial-omics-integrate",
        {
            "n_epochs": 0,
        },
    )

    assert "n_epochs" not in normalized


def test_build_skill_llm_cli_args_for_spatial_omics_integrate():
    cli_args = build_skill_llm_cli_args(
        "spatial-omics-integrate",
        {
            "method": "spaddm",
            "n_epochs": 1,
            "device": "gpu",
            "platform": "auto",
        },
    )

    assert cli_args == [
        "--method",
        "spaddm",
        "--n-epochs",
        "1",
        "--device",
        "gpu",
        "--data-type",
        "auto",
    ]


def test_build_skill_llm_cli_args_supports_skill_specific_platform_flag():
    cli_args = build_skill_llm_cli_args(
        "spatial-modality-integrate",
        {"platform": "Visium"},
    )

    assert cli_args == ["--platform", "Visium"]


def test_build_skill_llm_cli_args_maps_n_epochs_for_spatial_modality_integrate():
    cli_args = build_skill_llm_cli_args(
        "spatial-modality-integrate",
        {
            "method": "pearlst",
            "n_epochs": 1,
        },
    )

    assert cli_args == [
        "--method",
        "pearlst",
        "--epochs",
        "1",
    ]


def test_normalize_skill_llm_args_maps_pearlst_aliases():
    normalized = normalize_skill_llm_args(
        "spatial-modality-integrate",
        {
            "method": "PearIST",
            "platform": "slideseqv2",
        },
    )

    assert normalized["method"] == "pearlst"
    assert normalized["platform"] == "Slide-seqV2"


def test_spatial_cnv_registry_keeps_multi_value_reference_categories():
    registry.load_all()
    skill = registry.skills["spatial-cnv"]

    assert "--reference-cat" in skill["allowed_extra_flags"]
    assert "--window-size" in skill["allowed_extra_flags"]
    assert "--step" in skill["allowed_extra_flags"]
    assert skill["multi_value_extra_flags"] == {"--reference-cat"}


def test_spatial_velocity_registry_matches_script_flags():
    registry.load_all()
    skill = registry.skills["spatial-velocity"]

    assert "--method" in skill["allowed_extra_flags"]
    assert "--mode" not in skill["allowed_extra_flags"]


def test_spatial_deconvolution_registry_does_not_advertise_demo():
    registry.load_all()
    skill = registry.skills["spatial-deconvolution"]

    assert "demo_args" not in skill


def test_spatial_deconvolution_llm_method_schema_matches_supported_methods():
    registry.load_all()
    skill = registry.skills["spatial-deconvolution"]
    method_schema = skill["llm_argument_schema"]["method"]

    assert method_schema["allowed_values"] == ("tangram", "stereoscope", "graphst")
    assert "cell2location" not in method_schema["allowed_values"]
    assert "destvi" not in method_schema["allowed_values"]
    assert "flashdeconv" not in method_schema["allowed_values"]

    normalized = normalize_skill_llm_args(
        "spatial-deconvolution",
        {"method": "cell2location"},
    )
    assert "method" not in normalized

    normalized = normalize_skill_llm_args(
        "spatial-deconvolution",
        {"method": "graph st"},
    )
    cli_args = build_skill_llm_cli_args("spatial-deconvolution", normalized)
    assert cli_args == ["--method", "graphst"]


def test_registered_skill_extra_flags_match_argparse_contracts():
    registry.load_all()

    for skill_name, skill in registry.skills.items():
        script_path = Path(skill["script"])
        if not script_path.exists():
            continue

        script_extra_flags = _argparse_cli_flags(script_path) - _STANDARD_CLI_FLAGS
        registry_extra_flags = set(skill.get("allowed_extra_flags", set()))

        assert script_extra_flags <= registry_extra_flags, (
            skill_name,
            sorted(script_extra_flags - registry_extra_flags),
        )
        assert registry_extra_flags <= script_extra_flags, (
            skill_name,
            sorted(registry_extra_flags - script_extra_flags),
        )


def test_skill_markdown_cli_flags_match_argparse_contracts():
    registry.load_all()

    for skill_name, skill in registry.skills.items():
        script_path = Path(skill["script"])
        skill_md = script_path.parent / "SKILL.md"
        if not script_path.exists() or not skill_md.exists():
            continue

        script_flags = _argparse_cli_flags(script_path)
        documented_flags = _markdown_cli_flags(skill_md)

        assert script_flags <= documented_flags, (
            skill_name,
            sorted(script_flags - documented_flags),
        )
        assert documented_flags - {"--help"} <= script_flags, (
            skill_name,
            sorted((documented_flags - {"--help"}) - script_flags),
        )


def test_skill_markdown_documents_argparse_choices_and_defaults():
    registry.load_all()

    for skill_name, skill in registry.skills.items():
        script_path = Path(skill["script"])
        skill_md = script_path.parent / "SKILL.md"
        if not script_path.exists() or not skill_md.exists():
            continue

        contracts = _argparse_argument_contracts(script_path)
        markdown = _markdown_text(skill_md)

        for flag, contract in contracts.items():
            choices = contract.get("choices")
            if choices and choices != "DYNAMIC":
                missing_choices = [
                    choice
                    for choice in choices
                    if str(choice) not in markdown
                ]
                assert not missing_choices, (skill_name, flag, missing_choices)

            default = contract.get("default")
            if flag in _STANDARD_CLI_FLAGS or default in (None, False, "DYNAMIC"):
                continue
            assert str(default) in markdown or f"`{default}`" in markdown, (
                skill_name,
                flag,
                default,
            )


def test_llm_schema_choices_match_argparse_choices_when_declared():
    registry.load_all()

    for skill_name, skill in registry.skills.items():
        script_path = Path(skill["script"])
        if not script_path.exists():
            continue

        contracts = _argparse_argument_contracts(script_path)
        for canonical, schema in (skill.get("llm_argument_schema") or {}).items():
            cli_flag = schema.get("cli_flag")
            allowed_values = schema.get("allowed_values")
            if not cli_flag or not allowed_values or cli_flag not in contracts:
                continue
            choices = contracts[cli_flag].get("choices")
            if choices in (None, "DYNAMIC"):
                continue
            assert tuple(choices) == tuple(allowed_values), (
                skill_name,
                canonical,
                choices,
                allowed_values,
            )


def test_user_visible_skill_contract_text_has_no_retired_interfaces():
    user_visible_paths = [
        Path("bot/core.py"),
        Path("spatialclaw/agents/tools.py"),
        Path("spatialclaw/core/registry.py"),
        Path("skills/spatial/spatial-orchestrator/spatial_orchestrator.py"),
        Path("docs/METHODS.md"),
        Path("pyproject.toml"),
        *Path("skills/spatial").glob("*/SKILL.md"),
    ]
    retired_patterns = {
        "old_deconvolution_methods": re.compile(
            r"cell2location|flashdeconv|destvi",
            re.IGNORECASE,
        ),
        "old_or_invalid_flags": re.compile(
            r"--use-image|--root-cell-type|--no-filter-markers|--source|--method gsea|--method ssgsea"
        ),
        "skill_homepage_field": re.compile(r"homepage:"),
    }

    hits = []
    for path in user_visible_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in retired_patterns.items():
            if pattern.search(text):
                hits.append((label, str(path)))

    assert not hits
