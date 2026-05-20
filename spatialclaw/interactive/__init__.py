"""SpatialClaw interactive CLI/TUI package.

Entry points:
    spatialclaw interactive   — Rich CLI with prompt_toolkit REPL
    spatialclaw tui           — Textual full-screen TUI
    spatialclaw --ui tui      — Same, via flag
"""

from .interactive import run_interactive  # noqa: F401


def main() -> None:
    """Default CLI entry point — starts interactive mode."""
    run_interactive()
