# AGENTS.md — SpatialClaw Guide for AI Coding Agents

This guide is for AI coding agents working on the SpatialClaw codebase.

## Project Overview

SpatialClaw is a spatial transcriptomics analysis platform. The current
architecture is spatial-only: non-spatial omics skills and cross-domain
orchestrators have been removed. Each remaining skill is a self-contained
module that runs through the CLI or Python APIs, with local-first processing and
standardized reports.

The main package is `spatialclaw/`, and the root script is `spatialclaw.py`.
Installed commands are `spatialclaw`, `sc`, `spatialclaw-chat`, and `sc-chat`.

## Setup

```bash
cd /lanou/Claw/SpatialClaw
pip install -e .

python spatialclaw.py list
python spatialclaw.py run spatial-preprocessing --demo
spatialclaw list
sc list
spatialclaw run spatial-preprocessing --demo
sc run spatial-preprocessing --demo
```

## Commands

| Command | Purpose |
|---------|---------|
| `python spatialclaw.py list` | List registered spatial skills |
| `python spatialclaw.py run <canonical-skill> --demo` | Run a skill demo when that skill registers `demo_args` |
| `python spatialclaw.py run <canonical-skill> --input <file> --output <dir>` | Run with user data |
| `python spatialclaw.py interactive` | Start interactive terminal chat |
| `python spatialclaw.py interactive --ui tui` | Start full-screen Textual TUI |
| `python spatialclaw.py interactive -p "<prompt>"` | Single-shot LLM/tool mode |
| `spatialclaw-chat -p "<prompt>"` | Direct chat alias for `spatialclaw interactive -p` |
| `sc-chat -p "<prompt>"` | Short direct chat alias |
| `sc list` | Short alias for `spatialclaw list` |
| `python spatialclaw.py interactive --session <id>` | Resume a previous session |
| `python spatialclaw.py tui` | Start the TUI |
| `python spatialclaw.py mcp list` | List configured MCP servers |
| `python spatialclaw.py mcp add <name> <cmd> [args]` | Add an MCP server |
| `python spatialclaw.py mcp remove <name>` | Remove an MCP server |
| `python spatialclaw.py mcp config` | Show MCP config file path |
| `python spatialclaw.py onboard` | Run interactive setup wizard |
| `python spatialclaw.py memory-server` | Start graph memory REST API |
| `python -m pytest -v` | Run tests |
| `make demo` | Run the spatial preprocessing demo |
| `make bot-telegram` | Start Telegram bot |
| `make bot-feishu` | Start Feishu bot |

Short skill aliases are not supported. Use canonical registry names such as
`spatial-preprocessing`, `spatial-domain-identification`,
`spatial-svg-detection`, `spatial-cell-annotation`,
`spatial-deconvolution`, `spatial-cell-communication`,
`spatial-condition-comparison`, `spatial-integration`, and
`spatial-registration`.

## Project Structure

```text
SpatialClaw/
├── spatialclaw.py                  # Root CLI script
├── spatialclaw/                    # Core framework
│   ├── common/                     # report.py, session.py, checksums.py
│   ├── core/                       # registry.py, dependency_manager.py
│   ├── loaders/                    # Spatial data loaders
│   ├── memory/                     # Graph memory system
│   ├── routing/                    # LLM routing helpers
│   ├── agents/                     # Research agent modules
│   └── interactive/                # CLI/TUI interactive interface
├── skills/
│   ├── orchestrator/              # Top-level spatial orchestration wrapper
│   └── spatial/                    # Spatial skills and shared utilities
│       ├── _lib/                   # Shared spatial utilities
│       ├── spatial-preprocessing/
│       ├── spatial-domain-identification/
│       ├── spatial-cell-annotation/
│       ├── spatial-deconvolution/
│       ├── spatial-modality-integrate/
│       └── spatial-orchestrator/
├── bot/                            # Messaging bot frontends and shared LLM loop
├── templates/                      # Skill templates
├── examples/                       # Demo/sample data
├── sessions/                       # SpatialSession JSON files
├── docs/
└── results/
```

Import convention: spatial utilities are imported via
`from skills.spatial._lib.<module> import <name>`. The `_lib/` directory is an
internal shared package and is not registered as a skill.

## Skill Architecture

Every skill has a `SKILL.md` with YAML frontmatter and methodology, plus a
Python script accepting `--input`, `--output`, `--demo` when applicable, and
skill-specific flags declared in `spatialclaw/core/registry.py`.

Skills are registered as canonical names in `SpatialSkillRegistry`. Dynamic
discovery is spatial-only and ignores directories starting with `_`, `.`, or
`__`.

## How to Add a Spatial Skill

1. Create `skills/spatial/<skill-directory>/`.
2. Add `SKILL.md`.
3. Add a Python script with a standard CLI.
4. Add focused tests.
5. Register the canonical skill name and allowed flags in `spatialclaw/core/registry.py` when stable.
6. Regenerate the catalog if needed: `python scripts/generate_catalog.py`.

## Graph Memory System

SpatialClaw uses a graph-based memory system in `spatialclaw/memory/` to persist
context across sessions, agents, and tool invocations. Memory is backed by
SQLite/PostgreSQL through SQLAlchemy.

Important components:
- `MemoryClient`: high-level `remember()`, `recall()`, and `search()` API.
- `LayeredMemoryStore`: typed short-term / long-term API for dataset,
  analysis, preference, insight, and project-context records.
- `server.py`: FastAPI management API for `python spatialclaw.py memory-server`.

The obsolete compatibility memory module has been removed; new code should use
typed memory records from `spatialclaw.memory.types` through `LayeredMemoryStore`.

## Bot Integration

Messaging bot frontends live in `bot/`. Both Telegram and Feishu import
`bot/core.py`, which provides the shared LLM tool loop, skill execution,
security helpers, and audit logging.

```bash
pip install -r bot/requirements.txt
python bot/telegram_bot.py
python bot/feishu_bot.py
```

## Interactive CLI/TUI

```bash
python spatialclaw.py interactive
spatialclaw-chat
sc-chat
python spatialclaw.py interactive --ui tui
python spatialclaw.py interactive -p "run spatial-preprocessing demo"
sc-chat -p "run spatial-preprocessing demo"
python spatialclaw.py interactive --session <session-id>
python spatialclaw.py interactive --provider deepseek --model deepseek-chat
python spatialclaw.py interactive --workspace /path/to/workdir
python spatialclaw.py interactive --mode daemon
python spatialclaw.py interactive --mode run --name my-analysis
python spatialclaw.py tui
```

Slash commands include `/skills`, `/run`, `/sessions`, `/resume`, `/delete`,
`/current`, `/new`, `/clear`, `/mcp`, `/config`, `/help`, and `/exit`.

## Safety Boundaries

1. Local-first: do not upload user data.
2. Every report must include the SpatialClaw disclaimer.
3. Do not invent unsupported science or parameters.
4. CLI passthrough flags must remain registry-whitelisted.
5. Use canonical skill names only.
