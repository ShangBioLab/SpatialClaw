#!/usr/bin/env python3
"""SpatialClaw — spatial omics analysis skills runner.

Usage:
    python spatialclaw.py list
    python spatialclaw.py run <skill> --demo   # only for skills that provide demo data
    python spatialclaw.py run <skill> --input <data> --output <dir>
    python spatialclaw.py run spatial-orchestrator --pipeline standard --input <h5ad> --output <dir>
    python spatialclaw.py upload --input <data> --data-type <type>
    spatialclaw list
    sc list

Interactive CLI/TUI:
    python spatialclaw.py interactive               # Rich CLI (prompt_toolkit)
    python spatialclaw.py interactive --ui tui      # Full-screen Textual TUI
    python spatialclaw.py interactive -p "..."      # Single-shot mode
    python spatialclaw.py interactive --session <id> # Resume session
    python spatialclaw.py tui                       # Alias for --ui tui
    spatialclaw-chat -p "..."                       # Direct chat alias
    sc-chat -p "..."                                # Short direct chat alias

MCP Server Management:
    python spatialclaw.py mcp list
    python spatialclaw.py mcp add <name> <command> [args]
    python spatialclaw.py mcp remove <name>
    python spatialclaw.py mcp config
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from spatialclaw.paths import (
    EXAMPLES_DIR,
    OUTPUT_DIR,
    PACKAGE_DIR,
    PROJECT_ROOT,
    SESSIONS_DIR,
    SKILLS_DIR,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_ROOT = OUTPUT_DIR
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

_COLOUR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
BOLD = "\033[1m" if _COLOUR else ""
DIM = "\033[2m" if _COLOUR else ""
GREEN = "\033[32m" if _COLOUR else ""
YELLOW = "\033[33m" if _COLOUR else ""
BLUE = "\033[34m" if _COLOUR else ""
MAGENTA = "\033[35m" if _COLOUR else ""
RED = "\033[31m" if _COLOUR else ""
CYAN = "\033[36m" if _COLOUR else ""
RESET = "\033[0m" if _COLOUR else ""

# ---------------------------------------------------------------------------
# Skills and Domain metadata registry
# ---------------------------------------------------------------------------

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spatialclaw.core.registry import registry
registry.load_all()
SKILLS = registry.skills
DOMAINS = registry.domains

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_extra_arg_value(extra_args: list[str] | None, flag: str) -> str | None:
    """Return the value for a CLI flag from a flat extra_args list."""
    if not extra_args:
        return None

    for i, arg in enumerate(extra_args):
        if arg == flag and i + 1 < len(extra_args):
            return extra_args[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def has_required_extra_inputs(extra_args: list[str] | None, required_specs: list[dict]) -> bool:
    """Return True when all registry-declared required extra inputs are present."""
    return all(extract_extra_arg_value(extra_args, spec["flag"]) for spec in required_specs)


def has_alternative_inputs(extra_args: list[str] | None, alternative_flags: set[str]) -> bool:
    """Return True when any registry-declared alternative input flag is present."""
    return any(extract_extra_arg_value(extra_args, flag) for flag in alternative_flags)


def has_input_optional_mode(extra_args: list[str] | None, mode_flags: set[str]) -> bool:
    """Return True when a registry-declared no-input operation mode is present."""
    if not extra_args or not mode_flags:
        return False
    for arg in extra_args:
        token = str(arg)
        if token in mode_flags:
            return True
        if token.startswith("--") and "=" in token:
            flag, _value = token.split("=", 1)
            if flag in mode_flags:
                return True
    return False


def _looks_like_negative_number(token: str) -> bool:
    if not isinstance(token, str) or not token.startswith("-") or token == "-":
        return False
    try:
        float(token)
    except ValueError:
        return False
    return True


def _is_cli_flag_token(token: str) -> bool:
    return isinstance(token, str) and token.startswith("-") and not _looks_like_negative_number(token)


def _normalize_cli_flag_name(flag: str) -> str:
    if not isinstance(flag, str) or not flag.startswith("-"):
        return flag
    normalized = flag.replace("_", "-")
    if normalized == "--cpu":
        return "--no-gpu"
    return normalized


def _consume_extra_arg_values(
    extra_args: list[str],
    start_index: int,
    *,
    multi_value: bool,
) -> tuple[list[str], int]:
    values: list[str] = []
    i = start_index
    while i < len(extra_args):
        token = str(extra_args[i])
        if _is_cli_flag_token(token):
            break
        values.append(token)
        i += 1
        if not multi_value:
            break
    return values, i


def _get_skill_path_extra_flags(skill_info: dict[str, Any]) -> set[str]:
    path_flags = set(skill_info.get("path_extra_flags", set()))
    for spec in skill_info.get("required_extra_inputs", []):
        if spec.get("kind") in {"file", "directory"}:
            path_flags.add(spec["flag"])
    path_flags.update(skill_info.get("alternative_input_flags", set()))
    return path_flags


def _prepare_skill_extra_args(
    skill_info: dict[str, Any],
    extra_args: list[str] | None,
) -> tuple[list[str], str | None]:
    """Normalize, validate, and path-resolve passthrough CLI arguments."""
    if not extra_args:
        return [], None

    allowed = set(skill_info.get("allowed_extra_flags", set()))
    blocked = {"--input", "--output", "--demo"}
    path_flags = _get_skill_path_extra_flags(skill_info)
    multi_value_flags = set(skill_info.get("multi_value_extra_flags", set()))
    normalized_args: list[str] = []
    invalid_tokens: list[str] = []
    i = 0

    while i < len(extra_args):
        raw_token = str(extra_args[i])
        if raw_token == "--":
            i += 1
            continue
        if not _is_cli_flag_token(raw_token):
            invalid_tokens.append(raw_token)
            i += 1
            continue

        inline_value: str | None = None
        if raw_token.startswith("--") and "=" in raw_token:
            raw_flag, inline_value = raw_token.split("=", 1)
            flag = _normalize_cli_flag_name(raw_flag)
        else:
            flag = _normalize_cli_flag_name(raw_token)

        expects_multiple = flag in multi_value_flags
        next_index = i + 1
        values: list[str] = []
        if inline_value is not None:
            values.append(inline_value)
            if expects_multiple:
                trailing_values, next_index = _consume_extra_arg_values(
                    extra_args,
                    next_index,
                    multi_value=True,
                )
                values.extend(trailing_values)
        else:
            values, next_index = _consume_extra_arg_values(
                extra_args,
                next_index,
                multi_value=expects_multiple,
            )

        if flag in blocked:
            i = next_index
            continue

        if flag not in allowed:
            invalid_tokens.append(flag)
            i = next_index
            continue

        if expects_multiple and not values:
            invalid_tokens.append(f"{flag} (requires one or more values)")
            i = next_index
            continue

        normalized_args.append(flag)
        for value in values:
            if flag in path_flags:
                value = str(Path(value).expanduser().resolve())
            normalized_args.append(value)

        i = next_index

    if invalid_tokens:
        allowed_display = ", ".join(sorted(allowed)) if allowed else "none"
        invalid_display = ", ".join(dict.fromkeys(invalid_tokens))
        skill_label = skill_info.get("alias", "skill")
        return [], (
            f"Unsupported extra argument(s) for '{skill_label}': {invalid_display}. "
            f"Allowed skill-specific flags: {allowed_display}."
        )

    return normalized_args, None


def list_skills(domain_filter: str | None = None) -> dict:
    """Print available spatial skills and return the registry mapping."""
    print(f"\n{BOLD}SpatialClaw Skills{RESET}")
    if domain_filter:
        print(f"{BOLD}{'=' * 60}{RESET}")
        print(f"Filtering by domain: {CYAN}{domain_filter}{RESET}\n")
    else:
        print(f"{BOLD}{'=' * 60}{RESET}\n")

    # 1. 按 domain 分组构建索引
    domain_skills: dict[str, list[tuple[str, dict]]] = {}
    for alias, info in SKILLS.items():
        d = info.get("domain", "other")
        domain_skills.setdefault(d, []).append((alias, info))

    # 2. 按 DOMAINS 中定义的顺序依次输出
    for domain_key, domain_info in DOMAINS.items():
        if domain_filter and domain_key != domain_filter:
            continue
        skills_in_domain = domain_skills.get(domain_key, [])
        if not skills_in_domain:
            continue

        domain_name = domain_info.get("name", domain_key.title())
        data_types = domain_info.get("primary_data_types", [])
        types_str = ", ".join(f".{t}" if t != "*" else "*" for t in data_types)

        # 领域标题
        print(f"{BOLD}{YELLOW}📂 {domain_name}{RESET}  "
              f"{CYAN}[{types_str}]{RESET}")
        print(f"   {'─' * 54}")

        for alias, info in skills_in_domain:
            script = info["script"]
            status = f"{GREEN}ready{RESET}" if script.exists() else f"{YELLOW}planned{RESET}"
            desc = info.get("description", "")
            print(f"   {CYAN}{alias:<18}{RESET} [{status}] {desc}")

        print()

    # 3. 展示未在 DOMAINS 中注册的动态发现技能
    known_domains = set(DOMAINS.keys())
    extra = [(a, i) for a, i in SKILLS.items() if i.get("domain", "other") not in known_domains]
    if extra:
        print(f"{BOLD}{YELLOW}📂 Other (Dynamically Discovered){RESET}")
        print(f"   {'─' * 54}")
        for alias, info in extra:
            script = info["script"]
            status = f"{GREEN}ready{RESET}" if script.exists() else f"{YELLOW}planned{RESET}"
            desc = info.get("description", "")
            print(f"   {CYAN}{alias:<18}{RESET} [{status}] {desc}")
        print()

    total = len(SKILLS)
    print(f"{BOLD}Total: {total} spatial skills{RESET}\n")
    return SKILLS


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def upload_session(
    input_path: str,
    data_type: str = "generic",
    species: str = "human",
) -> dict:
    """Create a SpatialSession from a spatial data file."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from spatialclaw.common.session import SpatialSession

    session = SpatialSession.from_file(
        input_path, data_type=data_type, species=species,
    )
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sid = session.metadata["session_id"]
    session_path = SESSIONS_DIR / f"{sid}.json"
    session.save(session_path)
    return {
        "success": True,
        "session_path": str(session_path),
        "session_id": sid,
        "data_type": data_type,
    }


# ---------------------------------------------------------------------------
# Skill execution
# ---------------------------------------------------------------------------


def run_skill(
    skill_name: str,
    *,
    input_path: str | None = None,
    output_dir: str | None = None,
    demo: bool = False,
    session_path: str | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """Run a single skill via subprocess (waits until completion)."""

    skill_info = SKILLS.get(skill_name)
    if skill_info is None:
        return _err(skill_name, f"Unknown skill '{skill_name}'. Available: {list(SKILLS.keys())}")

    script_path: Path = skill_info["script"]
    if not script_path.exists():
        return _err(skill_name, f"Script not found: {script_path}")

    normalized_extra_args, extra_arg_error = _prepare_skill_extra_args(skill_info, extra_args)
    if extra_arg_error:
        return _err(skill_name, extra_arg_error)
    extra_args = normalized_extra_args or None

    # Resolve input from session if needed
    resolved_input = input_path
    if session_path and not input_path and not demo:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from spatialclaw.common.session import SpatialSession
        session = SpatialSession.load(session_path)
        if session.primary_data_path:
            resolved_input = session.primary_data_path

    # Resolve input to absolute path so subprocess cwd doesn't matter
    if resolved_input:
        resolved_input = str(Path(resolved_input).resolve())

    # Output directory
    if output_dir:
        out_dir = Path(output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUTPUT_ROOT / f"{skill_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [PYTHON, str(script_path)]
    supports_demo = bool(skill_info.get("demo_args"))
    required_extra_inputs = skill_info.get("required_extra_inputs", [])
    alternative_input_flags = set(skill_info.get("alternative_input_flags", set()))
    input_optional_flags = set(skill_info.get("input_optional_flags", set()))
    allows_alternative_inputs = bool(alternative_input_flags) or skill_info.get("allows_alternative_inputs", False)
    has_required_extras = has_required_extra_inputs(extra_args, required_extra_inputs)
    has_alternative_extras = has_alternative_inputs(extra_args, alternative_input_flags)
    has_input_optional_mode_flag = has_input_optional_mode(extra_args, input_optional_flags)
    if demo:
        if not supports_demo:
            return _err(
                skill_name,
                "Demo mode is not available for this skill. Use --input or its supported alternative input parameters.",
            )
        cmd.extend(skill_info["demo_args"])
    elif required_extra_inputs:
        required_flags = ", ".join(spec["flag"] for spec in required_extra_inputs)
        if not resolved_input:
            return _err(
                skill_name,
                f"This skill requires --input plus {required_flags}.",
            )
        if not has_required_extras:
            return _err(
                skill_name,
                f"Missing required extra input(s). Provide --input plus {required_flags}.",
            )
        cmd.extend(["--input", str(resolved_input)])
    elif resolved_input:
        cmd.extend(["--input", str(resolved_input)])
    elif allows_alternative_inputs and has_alternative_extras:
        # ✅ Skill supports alternative input modes (e.g., --input-list)
        # Extra args will be processed below; no --input needed
        pass
    elif has_input_optional_mode_flag:
        # Skills such as the orchestrator support query/list modes that do not
        # require an input dataset.
        pass
    else:
        expected_inputs = ["--input", "--session"]
        if supports_demo:
            expected_inputs.insert(1, "--demo")
        if required_extra_inputs:
            expected_inputs.extend(spec["flag"] for spec in required_extra_inputs)
        if allows_alternative_inputs:
            expected_inputs.extend(sorted(alternative_input_flags) or ["alternative input parameters"])
        if input_optional_flags:
            expected_inputs.extend(sorted(input_optional_flags))
        return _err(
            skill_name,
            f"No valid input provided. Use one of: {', '.join(expected_inputs)}.",
        )

    cmd.extend(["--output", str(out_dir)])

    # Print execution info with domain
    domain = skill_info.get("domain", "unknown")
    domain_display = DOMAINS.get(domain, {}).get("name", domain.title())
    if demo:
        mode_str = f"{CYAN}demo mode{RESET}"
    elif resolved_input:
        mode_str = f"input: {resolved_input}"
    elif has_input_optional_mode_flag:
        mode_str = f"{CYAN}input-optional mode{RESET}"
    else:
        mode_str = f"{CYAN}alternative inputs{RESET}"
    print(f"\n{BOLD}Running {domain_display} skill:{RESET} {GREEN}{skill_name}{RESET} ({mode_str})")
    print(f"{BOLD}Output:{RESET} {out_dir}\n")

    if extra_args:
        cmd.extend(extra_args)

    # Execute
    t0 = time.time()
    try:
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(script_path.parent),
            env=env,
        )
    except Exception as e:
        duration = time.time() - t0
        return _err(skill_name, str(e), duration=duration)

    duration = time.time() - t0

    # Collect output files
    output_files = sorted(
        [f.name for f in out_dir.rglob("*") if f.is_file()]
    ) if out_dir.exists() else []

    result = {
        "skill": skill_name,
        "success": proc.returncode == 0,
        "exit_code": proc.returncode,
        "output_dir": str(out_dir),
        "files": output_files,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_seconds": round(duration, 2),
    }

    # Update session if provided
    if session_path and result["success"]:
        _store_result_in_session(session_path, skill_name, out_dir)

    return result


def _store_result_in_session(
    session_path: str, skill_name: str, out_dir: Path,
) -> None:
    """Store skill result back into the session JSON."""
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from spatialclaw.common.session import SpatialSession

        result_json = out_dir / "result.json"
        if not result_json.exists():
            return
        session = SpatialSession.load(session_path)
        result_data = json.loads(result_json.read_text())
        session.add_skill_result(skill_name, result_data, output_dir=str(out_dir))

        processed = out_dir / "processed.h5ad"
        if processed.exists():
            session.primary_data_path = str(processed)
            session.mark_step(skill_name)

        session.save(session_path)
    except Exception:
        pass


def _err(skill: str, msg: str, duration: float = 0) -> dict:
    return {
        "skill": skill,
        "success": False,
        "exit_code": -1,
        "output_dir": None,
        "files": [],
        "stdout": "",
        "stderr": msg,
        "duration_seconds": round(duration, 2),
    }


# ---------------------------------------------------------------------------
# Workspace mode helpers (inspired by EvoScientist --mode / --name design)
# ---------------------------------------------------------------------------

RUNS_DIR = DEFAULT_OUTPUT_ROOT / "runs"


def _deduplicate_run_name(name: str, runs_dir: Path | None = None) -> str:
    """Return *name* if available, otherwise *name_1*, *name_2*, etc."""
    if runs_dir is None:
        runs_dir = RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not (runs_dir / name).exists():
        return name
    i = 1
    while (runs_dir / f"{name}_{i}").exists():
        i += 1
    return f"{name}_{i}"


def _resolve_workspace(
    workspace_dir: str | None,
    mode: str | None,
    run_name: str | None,
) -> str | None:
    """Resolve the effective workspace directory.

    - ``--workspace <dir>`` always wins (explicit override).
    - ``--mode daemon`` uses workspace_dir or project root (persistent).
    - ``--mode run`` creates an isolated ``output/runs/<name_or_ts>/`` dir.
    - ``--name`` gives the run directory a human-friendly name (only with run mode).
    """
    import os
    import re

    # Validate: --name only with --mode run
    if run_name and mode != "run":
        print(f"{RED}Error: --name can only be used with --mode run{RESET}",
              file=sys.stderr)
        sys.exit(1)

    # Sanitize run name
    if run_name and not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        print(f"{RED}Error: --name may only contain letters, digits, hyphens, and underscores{RESET}",
              file=sys.stderr)
        sys.exit(1)

    # Explicit --workspace always wins
    if workspace_dir:
        ws = os.path.abspath(os.path.expanduser(workspace_dir))
        os.makedirs(ws, exist_ok=True)
        return ws

    if mode == "run":
        if run_name:
            session_id = _deduplicate_run_name(run_name)
        else:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        ws = str(RUNS_DIR / session_id)
        os.makedirs(ws, exist_ok=True)
        return ws

    if mode == "daemon":
        # Daemon mode: use project root (persistent)
        return str(PROJECT_ROOT)

    # No mode specified: return None (let downstream use default)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class SpatialClawParser(argparse.ArgumentParser):
    """Custom parser for beautiful SpatialClaw CLI help output."""
    
    def print_help(self, file=None):
        if file is None:
            file = sys.stdout

        print(f"\n{BOLD}{CYAN}⬡ SpatialClaw{RESET} — AI-powered spatial omics platform\n", file=file)
        print(f"{BOLD}Usage:{RESET} spatialclaw <command> [options]", file=file)
        print(f"       sc <command> [options]", file=file)
        print(f"       spatialclaw-chat [interactive options]", file=file)
        print(f"       sc-chat [interactive options]\n", file=file)

        print(f"{BOLD}{YELLOW}🌟 Core Commands{RESET}", file=file)
        print(f"  {GREEN}interactive{RESET}  AI interactive terminal (CLI mode)", file=file)
        print(f"  {GREEN}tui        {RESET}  Advanced full-screen Textual interface", file=file)
        print(f"  {GREEN}list       {RESET}  List available spatial analysis skills", file=file)
        print(f"  {GREEN}run        {RESET}  Execute a specific skill (e.g., 'spatialclaw run spatial-preprocessing')", file=file)

        print(f"\n{BOLD}{BLUE}🔧 Utility Commands{RESET}", file=file)
        print(f"  {GREEN}mcp           {RESET}  Manage external Model Context Protocol (MCP) servers", file=file)
        print(f"  {GREEN}memory-server {RESET}  Start the graph memory REST API server", file=file)
        print(f"  {GREEN}env           {RESET}  Check installed Python dependencies and system tiers", file=file)
        print(f"  {GREEN}onboard       {RESET}  Interactive setup wizard to configure API keys", file=file)
        print(f"  {GREEN}upload        {RESET}  Upload/initialize session from existing .h5ad data", file=file)

        print(f"\n{BOLD}{MAGENTA}⚙  Global Options{RESET}", file=file)
        print(f"  {GREEN}-m, --mode {RESET}  Workspace mode: {CYAN}daemon{RESET} (persistent) | {CYAN}run{RESET} (isolated per-session)", file=file)
        print(f"  {GREEN}-n, --name {RESET}  Name for run session directory (requires --mode run)", file=file)
        print(f"  {GREEN}--workspace{RESET}  Override workspace directory for this session", file=file)

        print(f"\n{BOLD}For specific command help, use:{RESET} spatialclaw <command> --help\n", file=file)
        print(f"{DIM}Aliases: sc = spatialclaw; spatialclaw-chat/sc-chat = spatialclaw interactive.{RESET}", file=file)

        print(f"{DIM}SpatialClaw project is under active development.{RESET}\n", file=file)


def main():
    # Ensure .env is loaded for all subcommands (memory-server, etc.)
    try:
        from dotenv import load_dotenv as _load_dotenv
        _env_path = PROJECT_ROOT / ".env"
        if _env_path.exists():
            _load_dotenv(str(_env_path), override=False)
    except ImportError:
        pass

    parser = SpatialClawParser(
        description="SpatialClaw — spatial omics skills runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # list
    list_p = sub.add_parser("list", help="List available skills")
    list_p.add_argument("--domain", help="Filter by domain (only: spatial)")

    # env
    env_p = sub.add_parser("env", help="Check installed SpatialClaw dependency tiers")

    # upload
    upload_p = sub.add_parser("upload", help="Create a spatial session from h5ad data")
    upload_p.add_argument("--input", required=True, dest="input_path")
    upload_p.add_argument("--data-type", default="generic")
    upload_p.add_argument("--species", default="human")

    # onboard
    onboard_p = sub.add_parser("onboard", help="Run interactive setup wizard to configure API keys and channels")

    interactive_p = sub.add_parser("interactive", help="Start interactive terminal chat with LLM and skills")
    interactive_p.add_argument("--session", dest="session_id", default=None,
                               help="Resume a saved session by ID (or prefix)")
    interactive_p.add_argument("-p", "--prompt", dest="prompt", default=None,
                               help="Single-shot prompt (non-interactive, print response and exit)")
    interactive_p.add_argument("--ui", choices=["cli", "tui"], default="cli",
                               help="UI backend: cli (default, prompt_toolkit) or tui (Textual full-screen)")
    interactive_p.add_argument("--model", default="", help="Override LLM model name")
    interactive_p.add_argument("--provider", default="", help="Override LLM provider (deepseek, openai, gemini, ...)")
    interactive_p.add_argument("--workspace", dest="workspace_dir", default=None,
                               help="Working directory for this session (default: project root)")
    interactive_p.add_argument("-m", "--mode", dest="mode", default=None,
                               choices=["daemon", "run"],
                               help="Workspace mode: 'daemon' (persistent, default) or 'run' (isolated per-session)")
    interactive_p.add_argument("-n", "--name", dest="run_name", default=None,
                               help="Name for this run session (used as directory name; requires --mode run)")

    # tui
    tui_p = sub.add_parser("tui", help="Start advanced full-screen Textual User Interface")
    tui_p.add_argument("--session", dest="session_id", default=None,
                       help="Resume a saved session by ID")
    tui_p.add_argument("--model", default="", help="Override LLM model name")
    tui_p.add_argument("--provider", default="", help="Override LLM provider")
    tui_p.add_argument("--workspace", dest="workspace_dir", default=None,
                       help="Working directory for this session")
    tui_p.add_argument("-m", "--mode", dest="mode", default=None,
                       choices=["daemon", "run"],
                       help="Workspace mode: 'daemon' (persistent) or 'run' (isolated per-session)")
    tui_p.add_argument("-n", "--name", dest="run_name", default=None,
                       help="Name for this run session (requires --mode run)")

    # mcp — manage external MCP servers
    mcp_p = sub.add_parser("mcp", help="Manage external MCP (Model Context Protocol) servers")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command")
    # mcp list
    mcp_sub.add_parser("list", help="List configured MCP servers")
    # mcp add
    mcp_add_p = mcp_sub.add_parser("add", help="Add an MCP server")
    mcp_add_p.add_argument("name", help="Server name")
    mcp_add_p.add_argument("command", help="Command or URL")
    mcp_add_p.add_argument("args", nargs="*", help="Additional args for stdio transport")
    mcp_add_p.add_argument("--transport", choices=["stdio", "http", "sse", "websocket"], default=None)
    mcp_add_p.add_argument("--env", nargs="+", metavar="KEY=VAL", help="Environment variables")
    # mcp remove
    mcp_rm_p = mcp_sub.add_parser("remove", help="Remove an MCP server")
    mcp_rm_p.add_argument("name", help="Server name to remove")
    # mcp config — show config file path
    mcp_sub.add_parser("config", help="Show MCP config file path")

    # memory-server — start graph memory REST API
    mem_p = sub.add_parser("memory-server", help="Start the graph memory REST API server")
    mem_p.add_argument("--host", default=None, help="Host to bind (default: 0.0.0.0)")
    mem_p.add_argument("--port", type=int, default=None, help="Port to bind (default: 8766)")

    
    # run
    run_p = sub.add_parser(
        "run",
        help="Run a skill",
        description=(
            "Run a skill with generic input/output flags. "
            "Any additional skill-specific flags are passed through after parsing "
            "and validated against the selected skill's registry metadata."
        ),
        epilog=(
            "Examples:\n"
            "  spatialclaw run spatial-preprocessing --input sample.h5ad --output results --species human\n"
            "  spatialclaw run spatial-omics-integrate --input rna.h5ad --output results "
            "--method spatialglue --omics2 protein.h5ad\n"
            "  spatialclaw run spatial-modality-integrate --input /data/DLPFC/151673 --output results "
            "--method pearlst --ground-truth labels.tsv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    run_p.add_argument("skill", help="Canonical skill name (e.g. spatial-preprocessing, spatial-orchestrator)")
    run_p.add_argument("--demo", action="store_true")
    run_p.add_argument("--input", dest="input_path")
    run_p.add_argument("--output", dest="output_dir")
    run_p.add_argument("--session", dest="session_path")

    run_passthrough_args: list[str] = []
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        args, run_passthrough_args = parser.parse_known_args()
    else:
        args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "list":
        list_skills(domain_filter=getattr(args, "domain", None))
        sys.exit(0)

    if args.command == "onboard":
        from bot.onboard import run_onboard
        run_onboard()
        sys.exit(0)

    if args.command == "interactive":
        _mode = getattr(args, "mode", None) or "daemon"
        _run_name = getattr(args, "run_name", None)
        _ws = getattr(args, "workspace_dir", None)
        _ws = _resolve_workspace(_ws, _mode, _run_name)
        from spatialclaw.interactive.interactive import run_interactive
        run_interactive(
            workspace_dir=_ws,
            session_id=getattr(args, "session_id", None),
            model=getattr(args, "model", ""),
            provider=getattr(args, "provider", ""),
            ui_backend=getattr(args, "ui", "cli"),
            prompt=getattr(args, "prompt", None),
            mode=_mode,
            run_name=_run_name,
        )
        sys.exit(0)

    if args.command == "tui":
        _mode = getattr(args, "mode", None) or "daemon"
        _run_name = getattr(args, "run_name", None)
        _ws = getattr(args, "workspace_dir", None)
        _ws = _resolve_workspace(_ws, _mode, _run_name)
        from spatialclaw.interactive.interactive import run_interactive
        run_interactive(
            workspace_dir=_ws,
            session_id=getattr(args, "session_id", None),
            model=getattr(args, "model", ""),
            provider=getattr(args, "provider", ""),
            ui_backend="tui",
            mode=_mode,
            run_name=_run_name,
        )
        sys.exit(0)

    if args.command == "mcp":
        from spatialclaw.interactive._mcp import (
            list_mcp_servers,
            add_mcp_server,
            remove_mcp_server,
            MCP_CONFIG_PATH,
        )
        mcp_cmd = getattr(args, "mcp_command", None) or "list"
        if mcp_cmd == "list":
            servers = list_mcp_servers()
            if not servers:
                print(f"{YELLOW}No MCP servers configured.{RESET}")
                print(f"{CYAN}Add with: python spatialclaw.py mcp add <name> <command>{RESET}")
            else:
                print(f"\n{BOLD}MCP Servers{RESET}")
                print(f"{BOLD}{'=' * 50}{RESET}")
                for s in servers:
                    transport = s.get('transport', '?')
                    target = s.get('command') or s.get('url', '?')
                    print(f"  {CYAN}{s['name']:<20}{RESET} [{transport}] {target}")
            sys.exit(0)

        elif mcp_cmd == "add":
            env_dict: dict = {}
            for kv in (getattr(args, "env", None) or []):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    env_dict[k] = v
            try:
                entry = add_mcp_server(
                    args.name, args.command,
                    extra_args=args.args or None,
                    transport=getattr(args, "transport", None),
                    env=env_dict or None,
                )
                print(f"{GREEN}Added MCP server:{RESET} {args.name} ({entry['transport']})")
            except Exception as e:
                print(f"{RED}Error:{RESET} {e}", file=sys.stderr)
                sys.exit(1)
            sys.exit(0)

        elif mcp_cmd == "remove":
            from spatialclaw.interactive._mcp import remove_mcp_server
            if remove_mcp_server(args.name):
                print(f"{GREEN}Removed:{RESET} {args.name}")
            else:
                print(f"{RED}Not found:{RESET} {args.name}", file=sys.stderr)
                sys.exit(1)
            sys.exit(0)

        elif mcp_cmd == "config":
            print(f"MCP config file: {CYAN}{MCP_CONFIG_PATH}{RESET}")
            sys.exit(0)

        else:
            print(f"Usage: python spatialclaw.py mcp [list|add|remove|config]")
            sys.exit(1)

    if args.command == "memory-server":
        import os
        if getattr(args, "host", None):
            os.environ["SPATIALCLAW_MEMORY_HOST"] = args.host
        if getattr(args, "port", None):
            os.environ["SPATIALCLAW_MEMORY_PORT"] = str(args.port)
        from spatialclaw.memory.server import main as _mem_main
        _mem_main()
        sys.exit(0)

    if args.command == "env":
        from spatialclaw.core.dependency_manager import get_installed_tiers
        tiers = get_installed_tiers()
        
        print(f"\n{BOLD}SpatialClaw Environment Status{RESET}")
        print(f"{BOLD}{'=' * 40}{RESET}")
        
        core_status = f"{GREEN}✅ Installed{RESET}" if tiers.get("core") else f"{RED}❌ Missing{RESET}"
        print(f"Core System:      {core_status}")
        
        print(f"\n{BOLD}Domain Tiers:{RESET}")
        for tier in ["spatial"]:
            is_installed = tiers.get(tier, False)
            if is_installed:
                status = f"{GREEN}✅ Installed{RESET}"
            else:
                status = f"{RED}❌ Missing{RESET} (Run: pip install -e \".[{tier}]\")"
            print(f"- {tier.capitalize():<15} {status}")
            
        print(f"\n{BOLD}Spatial Skill Layers:{RESET}")
        standalone_layers = [
            ("Domain Identification", "spatial-domain-identification", "spatial-domain-identification", "Deep learning spatial domain methods, e.g., SpaGCN"),
            ("Cell Annotation",       "spatial-cell-annotation",       "spatial", "Cell type annotation, e.g., Tangram, scANVI"),
            ("Deconvolution",         "spatial-deconvolution",         "spatial", "Cell type deconvolution, e.g., Cell2Location, FlashDeconv"),
            ("Trajectory",            "spatial-trajectory",            "spatial", "Trajectory inference, e.g., CellRank, Palantir"),
            ("SVG Detection",         "spatial-svg-detection",         "spatial", "Spatially variable genes, e.g., SpatialDE"),
            ("Statistics",            "spatial-statistics",            "spatial", "Spatial statistics, e.g., Moran's I, Geary's C"),
            ("Condition Comparison",  "spatial-condition-comparison",  "spatial", "Condition comparison, e.g., PyDESeq2 pseudobulk"),
            ("Velocity",              "spatial-velocity",              "spatial", "RNA velocity analysis, e.g., scVelo, VeloVI"),
            ("CNV",                   "spatial-cnv",                   "spatial", "Copy number variation inference, e.g., inferCNVpy"),
            ("Enrichment",            "spatial-enrichment",            "spatial", "Pathway enrichment, e.g., GSEApy"),
            ("Cell Communication",    "spatial-cell-communication",    "spatial", "Cell communication, e.g., LIANA+, CellPhoneDB"),
            ("Integration",           "spatial-integration",           "spatial", "Multi-sample integration, e.g., Harmony, BBKNN"),
            ("Registration",          "spatial-registration",          "spatial", "Spatial registration, e.g., PASTE"),
        ]
        for label, tier_key, install_extra, desc in standalone_layers:
            sl_installed = tiers.get(tier_key, False)
            sl_status = f"{GREEN}✅ Installed{RESET}" if sl_installed else f"{RED}❌ Missing{RESET} (Run: pip install -e \".[{install_extra}]\")"
            print(f"- {label:<18} {sl_status} ({desc})")
        
        print(f"\nTo install all complete functionalities:\n  pip install -e \".[full]\"\n")
        sys.exit(0)

    if args.command == "upload":
        result = upload_session(
            args.input_path,
            data_type=args.data_type,
            species=args.species,
        )
        if result["success"]:
            print(f"{GREEN}Session created:{RESET} {result['session_path']}")
        else:
            print(f"{RED}Upload failed{RESET}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    if args.command == "run":
        extra = [str(token) for token in run_passthrough_args if token != "--"]
        result = run_skill(
            args.skill,
            input_path=args.input_path,
            output_dir=args.output_dir,
            demo=args.demo,
            session_path=args.session_path,
            extra_args=extra if extra else None,
        )

        if result["success"]:
            print(f"{GREEN}Success{RESET}: {result['skill']}")
            if result.get("output_dir"):
                print(f"  Output: {result['output_dir']}")
            if result.get("stdout"):
                print(result["stdout"], end="")
        else:
            print(f"{RED}Failed{RESET}: {result['skill']}", file=sys.stderr)
            if result.get("stderr"):
                print(result["stderr"], file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
