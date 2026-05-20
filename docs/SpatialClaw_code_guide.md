# SpatialClaw Code Guide

This guide describes the current spatial-only architecture after the cleanup
that removed non-spatial omics skills, short skill aliases, and obsolete
compatibility entrypoints.

## CLI Entrypoints

- `spatialclaw.py`: root script that delegates directly to `spatialclaw.cli_app.main`.
- `spatialclaw/cli.py`: package console-script entrypoint for `spatialclaw`.
- `spatialclaw/cli_app.py`: command parser, session upload, skill execution, and
  interactive/TUI/MCP/memory-server dispatch.

The CLI accepts canonical skill names only:

```bash
python spatialclaw.py run spatial-preprocessing --demo
python spatialclaw.py run spatial-domain-identification --input sample.h5ad --output results/domains
```

## Registry

`spatialclaw/core/registry.py` defines `SpatialSkillRegistry`, the spatial-only
skill registry. It exposes:

- `registry.skills`: canonical skill metadata.
- `registry.domains`: currently the single `spatial` domain.
- `registry.load_all()`: load hardcoded spatial skills plus CLI-capable spatial
  additions.
- `registry.load_lightweight()`: load `SKILL.md` frontmatter for spatial skills.
- `normalize_skill_llm_args()` and `build_skill_llm_cli_args()`: translate
  structured LLM tool arguments into whitelisted CLI flags.

Skill metadata includes `script`, `description`, `demo_args` when supported,
`allowed_extra_flags`, `path_extra_flags`, `multi_value_extra_flags`, and input
contract metadata.

## Sessions

`spatialclaw/common/session.py` contains `SpatialSession`. Sessions store:

- metadata such as session id, input file, data type, species, checksum, domain.
- `primary_data_path`, updated when a skill emits `processed.h5ad`.
- processing state and skill results.

New sessions are created with `SpatialSession.from_file(...)`.

## Skill Execution Flow

1. `cli_app.main()` parses the command.
2. `run_skill()` looks up the canonical skill name in `registry.skills`.
3. Extra arguments are normalized and checked against the skill's allowed flag set.
4. Input paths are resolved to absolute paths where needed.
5. The skill script runs in a subprocess with project root on `PYTHONPATH`.
6. Output files are collected and, if a session was provided, the session JSON is
   updated.

## Spatial Skills

Spatial skills live under `skills/spatial/`. Shared utilities live under
`skills/spatial/_lib/` and are imported explicitly:

```python
from skills.spatial._lib.adata_utils import ensure_spatial_neighbors
```

`_lib/` is not a skill and is ignored by registry discovery.

## Bot And Interactive Surfaces

`bot/core.py` contains the shared LLM tool loop and the `spatialclaw` tool
executor. Interactive CLI/TUI modules call into this same core so terminal and
bot sessions share routing behavior.

## Memory

The graph memory system remains in `spatialclaw/memory/`. It is intentionally
separate from spatial skill cleanup work unless a task explicitly asks for
memory refactoring.

## Adding Code

- Prefer existing spatial utility modules over new abstractions.
- Keep skill-specific dependencies lazy.
- Register new CLI flags in `SpatialSkillRegistry`.
- Add focused tests for new routing, validation, and skill behavior.
- Do not introduce short skill aliases or non-spatial skill trees.
