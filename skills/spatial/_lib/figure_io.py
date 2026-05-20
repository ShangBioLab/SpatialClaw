"""Figure output helpers shared by spatial skills."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib

logger = logging.getLogger(__name__)

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

matplotlib.use("Agg")

__all__ = ["non_interactive_backend", "save_figure"]


@contextmanager
def non_interactive_backend():
    """Context manager that ensures the Agg (non-interactive) backend is active."""
    prev = matplotlib.get_backend()
    matplotlib.use("Agg")
    try:
        yield
    finally:
        matplotlib.use(prev)


def _iter_figure_paths(output_dir: Path, filename: str) -> list[Path]:
    """Return one or more output paths for a figure.

    Default behavior is unchanged: save the requested filename only.
    Additional formats can be requested globally with
    ``SPATIALCLAW_EXTRA_FIGURE_FORMATS``, for example ``pdf`` or ``pdf,svg``.
    """
    base_path = output_dir / "figures" / filename
    paths = [base_path]

    extra_formats = os.getenv("SPATIALCLAW_EXTRA_FIGURE_FORMATS", "").strip()
    if not extra_formats:
        return paths

    seen = {base_path.suffix.lower()}
    for raw_fmt in extra_formats.split(","):
        fmt = raw_fmt.strip().lower().lstrip(".")
        if not fmt or not fmt.isalnum():
            logger.warning(
                "Ignoring invalid extra figure format %r from SPATIALCLAW_EXTRA_FIGURE_FORMATS",
                raw_fmt,
            )
            continue
        suffix = f".{fmt}"
        if suffix in seen:
            continue
        seen.add(suffix)
        paths.append(base_path.with_suffix(suffix))
    return paths


def save_figure(
    fig: plt.Figure | None,
    output_dir: str | Path,
    filename: str,
    *,
    dpi: int = 200,
    close: bool = True,
) -> Path:
    """Save a matplotlib figure to *<output_dir>/figures/<filename>*.

    Returns the path to the saved file.
    """
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = _iter_figure_paths(output_dir, filename)

    if fig is None:
        fig = plt.gcf()

    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure: %s", path)
    if close:
        plt.close(fig)
    return paths[0]
