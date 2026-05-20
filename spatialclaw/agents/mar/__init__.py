"""Vendored ReMemR1 runtime module for SpatialClaw-pri.

This package vendors the runnable ReMemR1 source tree under
``spatialclaw/agents/mar`` so SpatialClaw can keep MAR/ReMemR1 startup,
docs, and benchmark assets in the same repository.

Checkpoint weights are intentionally not vendored here. Use ``CKPT_PATH`` to
point the serving scripts at an external HuggingFace-format checkpoint.
"""

