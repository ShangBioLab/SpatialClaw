"""Backward-compatible ReMemR1 operator import shim.

The executable copied operator now lives in ``mar_operator.py``. This module
stays in place so existing imports do not break while new code can refer to
the copied MAR naming directly.
"""

from .mar_operator import (
    MAROperator,
    MemoryAugmentedReasoningOperator,
    MemoryCandidate,
    NO_MEMORY,
    NO_RECALLED_MEMORY,
    RecallEntry,
    ReMemReasoningOperator,
    TEMPLATE,
    TEMPLATE_FINAL,
)

__all__ = [
    "MAROperator",
    "MemoryAugmentedReasoningOperator",
    "MemoryCandidate",
    "NO_MEMORY",
    "NO_RECALLED_MEMORY",
    "RecallEntry",
    "ReMemReasoningOperator",
    "TEMPLATE",
    "TEMPLATE_FINAL",
]
