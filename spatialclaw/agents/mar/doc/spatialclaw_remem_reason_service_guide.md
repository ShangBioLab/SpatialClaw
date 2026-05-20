# ReMemR1 Memory Reasoning Service for SpatialClaw

## Purpose

This service exposes a ReMemR1-style reasoning endpoint for external agent systems such as SpatialClaw.

It is designed for the hybrid architecture:

- main assistant: GPT / closed API model
- memory store: external graph / session memory system
- ReMemR1 service: separate reasoning operator over retrieved memory chunks

## What This Service Does

The service wraps the ReMemR1 reasoning style into:

- `POST /remem_reason`

Input:

- a user query
- candidate memory chunks from another system

Internal behavior:

- iterates candidate memories chunk by chunk
- updates a working memory after each step
- can emit callback-style revisit decisions over prior working memory
- returns a final condensed reasoning summary for the downstream assistant

## File

- `taskutils/memory_eval/remem_reason_server.py`

## Upstream Dependency

This service does not serve model weights itself.

It expects another OpenAI-compatible model endpoint to already exist, for example:

- vLLM OpenAI server
- SGLang OpenAI-compatible server

That upstream model should be your deployed RL-trained ReMemR1 checkpoint if you want to preserve ReMemR1's RL-specific behavior.

## Environment Variables

### Service binding

```bash
export REMEM_REASON_HOST=127.0.0.1
export REMEM_REASON_PORT=8765
```

### Upstream model service

```bash
export REMEM_UPSTREAM_BASE_URL=http://127.0.0.1:8000/v1
export REMEM_UPSTREAM_API_KEY=123-abc
export REMEM_UPSTREAM_MODEL=ReMemR1-3B
```

### Optional service auth

```bash
export REMEM_REASON_API_TOKEN=your-internal-token
```

### Optional defaults

```bash
export REMEM_REASON_MAX_CHUNKS=8
export REMEM_REASON_MAX_MEMORY_TOKENS=768
export REMEM_REASON_MAX_SUMMARY_TOKENS=1024
export REMEM_REASON_MAX_CHUNK_CHARS=1600
export REMEM_REASON_TEMPERATURE=0.0
export REMEM_REASON_TOP_P=1.0
```

## Launch

```bash
python taskutils/memory_eval/remem_reason_server.py
```

### Recommended real deployment path

If you want to preserve ReMemR1's RL-specific behavior, do not point this service at the base Qwen model.

Instead:

1. Train ReMemR1 in the `vllm` conda environment
2. Merge the actor checkpoint
3. Serve the merged `hf_ckpt` with vLLM
4. Point `remem_reason_server.py` to that vLLM endpoint

Example:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
cd /dugaoyuan/AgentAndClaw/Agents/ReMemR1-main

bash scripts/merge_ckpt.sh "results/memory_agent/ReMemR1_3B/global_step_200/actor"

export CKPT_PATH=/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main/results/memory_agent/ReMemR1_3B/global_step_200/actor/hf_ckpt
export SERVER_PYTHON=/dugaoyuan/.conda/envs/vllm/bin/python3
export SERVED_MODEL_NAME=ReMemR1-3B-RL
export SERVE_PORT=8000
bash scripts/3_serve_merged_ckpt_vllm.sh
```

In another shell:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
cd /dugaoyuan/AgentAndClaw/Agents/ReMemR1-main

export SERVER_PYTHON=/dugaoyuan/.conda/envs/vllm/bin/python3
export REMEM_UPSTREAM_BASE_URL=http://127.0.0.1:8000/v1
export REMEM_UPSTREAM_API_KEY=123-abc
export REMEM_UPSTREAM_MODEL=ReMemR1-3B-RL
export REMEM_REASON_PORT=8765
bash scripts/4_serve_remem_reason_service.sh
```

Health check:

```bash
curl http://127.0.0.1:8765/healthz
```

## Request Format

```json
{
  "session_id": "cli:u1:c1",
  "query": "Continue the previous analysis and use the dataset we discussed earlier",
  "model": "ReMemR1-3B",
  "candidates": [
    {
      "id": "mem_1",
      "type": "dataset",
      "text": "file_path: data/brain.h5ad\nplatform: Visium\npreprocessing_state: clustered"
    },
    {
      "id": "mem_2",
      "type": "analysis",
      "text": "skill: spatial-preprocessing\nmethod: leiden\nstatus: completed"
    }
  ],
  "config": {
    "max_chunks": 8,
    "max_memory_tokens": 768,
    "max_summary_tokens": 1024
  }
}
```

## Response Format

```json
{
  "summary": "Use the prior Visium dataset and preserve the preprocessing lineage.",
  "used_memory_ids": ["mem_1", "mem_2"],
  "callbacks": ["previous preprocessing run"],
  "trace": {
    "session_id": "cli:u1:c1",
    "steps": 2,
    "step_trace": [
      {
        "step": 0,
        "candidate_id": "mem_1",
        "candidate_type": "dataset",
        "recall_query": null
      }
    ],
    "upstream_model": "ReMemR1-3B"
  }
}
```

## SpatialClaw-side Mapping

Set these in SpatialClaw:

```bash
export SPATIALCLAW_REMEM_ENABLED=true
export SPATIALCLAW_REMEM_BASE_URL=http://127.0.0.1:8000/v1
export SPATIALCLAW_REMEM_MODEL=ReMemR1-3B-RL
export SPATIALCLAW_REMEM_API_KEY=123-abc
export SPATIALCLAW_REMEM_REASON_ENDPOINT=http://127.0.0.1:8765/remem_reason
```

Notes:

- `SPATIALCLAW_REMEM_BASE_URL` / `MODEL` are still used by the local fallback operator path.
- `SPATIALCLAW_REMEM_REASON_ENDPOINT` activates the dedicated service endpoint path.

## Limitations of This Stub

This is a service stub, not a full retraining pipeline change.

It preserves the ReMemR1 reasoning pattern, but:

- input is memory chunks, not original long document chunks
- recall currently revisits prior working memory summaries, not a richer learned retriever
- no reward / RL training changes are included here

The next deeper step would be to adapt training data and reward design so the deployed RL model is explicitly optimized for memory-chunk reasoning rather than long-document QA only.

## Checkpoint Persistence

During RL training, raw checkpoints are written under:

```bash
results/memory_agent/<EXP_LOG_NAME>/
```

For deployment, the checkpoint you should preserve is the merged actor checkpoint:

```bash
results/memory_agent/<EXP_LOG_NAME>/global_step_<N>/actor/hf_ckpt
```

That `hf_ckpt` directory is the deployable HuggingFace-format checkpoint to serve with vLLM.
