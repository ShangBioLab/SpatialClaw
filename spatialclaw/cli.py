"""Package-level CLI entrypoint for SpatialClaw."""

from __future__ import annotations

import sys

from spatialclaw.cli_app import main as _main


def main() -> None:
    """Run the SpatialClaw CLI."""
    _main()


def chat_main() -> None:
    """Run the SpatialClaw interactive chat CLI directly."""
    sys.argv = [sys.argv[0], "interactive", *sys.argv[1:]]
    _main()
