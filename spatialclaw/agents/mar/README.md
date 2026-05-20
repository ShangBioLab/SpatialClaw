# MAR: Memory-Augmented Reasoning Operator

This directory contains the local MAR components used by SpatialClaw's memory
augmented reasoning path. In SpatialClaw, MAR is not presented as a standalone
paper contribution. It is used as an internal reasoning operator that helps the
agent revisit prior working memory, recover relevant session context, and pass a
concise reasoning summary to downstream tool-calling agents.

## Role in SpatialClaw

SpatialClaw stores analysis context, datasets, preferences, and intermediate
insights in its graph memory. The MAR operator sits between user input and the
execution agent:

1. It retrieves candidate memory records related to the current query.
2. It reasons over those records in chunks instead of relying on one linear
   context window.
3. It maintains a compact working memory and can request callback-style recall
   when earlier memory needs to be revisited.
4. It returns a short summary that the downstream SpatialClaw agent can use when
   planning or executing spatial omics tasks.

This design is useful for long-running spatial omics studies, where a user may
return to a previous sample, parameter choice, domain annotation, or figure
preference after many interactions.

## Method Background

The MAR design in SpatialClaw is inspired by ReMemR1:

> Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents

The central idea is to make memory revisitable rather than treating long context
as a single one-pass document scan. A model incrementally updates working memory
while processing retrieved chunks, and it can explicitly recall earlier memory
when later reasoning depends on prior evidence. SpatialClaw adapts this idea to
scientific workflow memory: datasets, prior analyses, user preferences, and
biological interpretation notes can be recovered and summarized before the agent
continues execution.

Reference:

- Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang
  Wang, and An Zhang. "Look Back to Reason Forward: Revisitable Memory for
  Long-Context LLM Agents." arXiv:2509.23040.

## Integration Notes

The active SpatialClaw integration entrypoint is:

- `spatialclaw/agents/mar_operator.py`

The operator delegates model inference to an OpenAI-compatible endpoint or a
dedicated reasoning service configured by environment variables such as
`SPATIALCLAW_MAR_BASE_URL`, `SPATIALCLAW_MAR_MODEL`, and
`SPATIALCLAW_MAR_API_KEY`.

## Attribution and License

This directory includes reference components derived from or inspired by the
ReMemR1 implementation. The included MAR code is distributed under the MIT
License; see `LICENSE` in this directory. ReMemR1 also acknowledges components
from MemAgent, which is licensed under Apache License 2.0.

Please cite the original ReMemR1 work when using or discussing the memory
augmented reasoning components.
