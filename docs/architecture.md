# Architecture

## Overview

SpatialClaw is a skill-based, local-first spatial transcriptomics analysis
framework. Each analysis capability is packaged as a self-contained skill with
method documentation, implementation, and tests.

```text
User input (CLI / interactive / bot)
        |
        v
spatialclaw.py / spatialclaw.cli_app
        |
        v
SpatialSkillRegistry
        |
        v
skills/spatial/<skill> subprocess
        |
        v
standard outputs: report.md, result.json, figures, tables, processed.h5ad
```

## Directory Structure

```text
SpatialClaw/
├── spatialclaw.py
├── spatialclaw/
│   ├── common/
│   ├── core/
│   ├── loaders/
│   ├── memory/
│   ├── routing/
│   ├── agents/
│   └── interactive/
├── skills/
│   └── spatial/
│       ├── _lib/
│       ├── spatial-preprocessing/
│       ├── spatial-domain-identification/
│       ├── spatial-cell-annotation/
│       ├── spatial-deconvolution/
│       ├── spatial-modality-integrate/
│       └── spatial-orchestrator/
├── bot/
├── docs/
├── examples/
├── tests/
└── sessions/
```

## Registry

`spatialclaw/core/registry.py` owns canonical skill metadata. The CLI does not
resolve short skill aliases. All execution surfaces must pass canonical skill
names such as `spatial-preprocessing` or `spatial-domain-identification`.

The registry also stores:

- script path
- description
- demo availability
- allowed extra flags
- path-valued extra flags
- multi-value flags
- input contracts
- LLM argument normalization schemas

## Skill Structure

```text
skills/spatial/<skill>/
├── SKILL.md
├── <skill_script>.py
└── tests/
```

Every CLI-capable skill accepts `--output`, accepts `--input` when it needs user
data, and may expose `--demo` if demo data is available.

## Data Flow

Most skills read and write AnnData files. When a skill emits `processed.h5ad`,
session execution updates `SpatialSession.primary_data_path` so the next skill
can use the latest processed file.

## Orchestrator

`skills/spatial/spatial-orchestrator/spatial_orchestrator.py` routes natural language
queries and file extensions to canonical spatial skill names. Named pipelines
also use canonical names and chain `processed.h5ad` between steps.

## Bot And Interactive Layers

`bot/core.py` provides the shared LLM tool loop and subprocess skill executor.
`spatialclaw/interactive/interactive.py` and `spatialclaw/interactive/tui.py`
reuse the bot core for CLI/TUI chat.

## Memory

Graph memory defaults to `<project-root>/.config/spatialclaw/memory.db`, so CLI,
bot, TUI, and server processes share the same local database even when launched
from a different current working directory. `SPATIALCLAW_MEMORY_DB_URL` remains
the explicit override for custom SQLite or PostgreSQL deployments.

## Security

`spatialclaw.cli_app.run_skill()` validates skill-specific passthrough flags
against registry metadata. Unsupported flags are rejected before the subprocess
is launched.

## Testing

```bash
python -m pytest -v
python spatialclaw.py run spatial-preprocessing --demo
python spatialclaw.py run spatial-orchestrator --demo
```
