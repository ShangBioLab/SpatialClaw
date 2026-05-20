# Installation Guide

## Quick Start

```bash
git clone <your-repo-url>
cd SpatialClaw
pip install -e .

python spatialclaw.py env
python spatialclaw.py list
python spatialclaw.py run spatial-preprocessing --demo
sc list
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.10 or 3.11 |
| RAM | 8 GB | 32 GB+ |
| Disk | 2 GB | 10 GB+ with full optional spatial stack |
| GPU | Optional | CUDA GPU for deep learning methods |

## Dependency Tiers

Core install:

```bash
pip install -e .
```

Optional spatial tiers:

```bash
pip install -e ".[spatial]"
pip install -e ".[spatial-domain-identification]"
pip install -e ".[full]"
```

Interactive and development tiers:

```bash
pip install -e ".[interactive]"
pip install -e ".[tui]"
pip install -e ".[memory]"
pip install -e ".[research]"
pip install -e ".[dev]"
```

## Using The Tsinghua PyPI Mirror

```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e ".[full]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Verification

For a fast installation check, use the same smoke target as CI-oriented local
validation:

```bash
python spatialclaw.py list
sc list
make test-smoke
```

To verify demo execution:

```bash
python spatialclaw.py run spatial-preprocessing --demo --output /tmp/spatialclaw_demo
python spatialclaw.py run spatial-orchestrator --demo --output /tmp/spatialclaw_orchestrator
spatialclaw-chat --help
sc-chat --help
```

Run the full pytest suite only when you need broad regression coverage:

```bash
python -m pytest -v
```

On this server, validation should be run inside the configured conda
environment:

```bash
conda run -n sppy310 python spatialclaw.py list
conda run -n sppy310 make test-smoke
```

## Troubleshooting

If a package is missing, install it into the active environment:

```bash
pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

For CUDA mismatches, install the correct PyTorch wheel before optional spatial
deep learning tiers.
