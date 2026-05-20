# ReMemR1 复现说明

## 1. 项目概览

`ReMemR1-main` 是一个面向长上下文 LLM Agent 的强化学习训练项目，核心是可回访记忆（revisitable memory）机制。

仓库主要包含三条路径：

- 数据处理：`scripts/0_run_data_process.sh`
- 训练入口：`scripts/1_run_train_ReMemR1_3B.sh`、`scripts/1_run_train_ReMemR1_7B.sh`
- 评测入口：`scripts/2_run_eval_ReMemR1.sh`

## 2. 当前机器上的现实约束

我在当前机器上确认到：

- 可见 GPU 为 `NVIDIA A800 80GB PCIe`
- 目前只看到单卡
- 仓库 README 中的官方训练默认是多机多卡 RL 训练
- 仓库评测脚本默认也按 `tp=4` 配置，且 `run_eval.py` 里 `ReMemR1_3B` / `ReMemR1_7B` 仍是 `CHECKPOINT_PATH`

因此，复现应分两层进行：

- 最小可验证复现：环境安装、依赖导入、评测数据处理、脚本静态检查
- 官方结果复现：准备训练数据、多卡训练、合并 checkpoint、再评测

## 3. Conda 环境建议

README 推荐使用 Python 3.11。

由于当前机器直接从 `conda` 远程源创建环境时出现网络/SSL 问题，更稳妥的方式是：

1. 基于本机已存在的 Python 3.11 conda 环境做离线克隆
2. 再补装 ReMemR1 缺失依赖

本次实际创建并验证可激活的环境名：

```bash
rememr1_env
```

本次采用的创建方式：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p /dugaoyuan/.conda/envs/rememr1_env
cp -al /dugaoyuan/.conda/envs/agentsociety/. /dugaoyuan/.conda/envs/rememr1_env/
conda activate rememr1_env
```

说明：

- 这里使用的是本机已有 Python 3.11 conda 环境的硬链接副本
- 我已经验证 `conda activate rememr1_env` 可正常生效
- 这样可以绕过当前机器上 `conda` 远程拉包失败的问题

如果后续网络恢复正常，也可以使用仓库 README 的标准方式新建一个更“原生”的环境：

```bash
conda create -y -n rememr1_py311 python=3.11
conda activate rememr1_py311
```

## 4. 依赖安装思路

仓库没有提供完整的 `requirements.txt` 或 `environment.yml`，需要以 README 为主，并结合代码中的实际导入进行补齐。

README 给出的核心安装命令如下：

```bash
pip install httpx==0.23.1 aiohttp -U ray[serve,default] vllm
pip install nltk pyyaml beautifulsoup4 html2text wonderwords tenacity fire
pip install vllm==0.9 --index-url https://download.pytorch.org/whl/cu126
pip install "sglang==0.4.6"
pip install hydra-core accelerate tensordict torchdata wandb "tensordict<=0.6.2"
```

结合代码导入，至少还需要确保这些包可用：

```bash
pip install datasets scikit-learn tqdm transformers
```

我已经在 `rememr1_env` 中补齐并验证可导入的核心依赖包括：

```bash
torch transformers ray sglang
httpx aiohttp nltk pyyaml beautifulsoup4 html2text wonderwords tenacity fire
hydra-core accelerate tensordict torchdata wandb datasets scikit-learn tqdm
```

建议按下面顺序安装，出问题时更容易定位：

```bash
pip install httpx==0.23.1 aiohttp ray[serve,default]
pip install nltk pyyaml beautifulsoup4 html2text wonderwords tenacity fire
pip install hydra-core accelerate tensordict<=0.6.2 torchdata wandb
pip install datasets scikit-learn tqdm transformers
pip install "sglang==0.4.6"
pip install vllm==0.9
```

如果需要开启代理再下载：

```bash
source /dugaoyuan/clash/start.sh
```

补充说明：

- `sglang==0.4.6` 已安装成功
- `vllm==0.9` 在当前机器上会触发一次非常重的 `torch/CUDA` 依赖替换，暂未继续完成
- 如果后续要走官方训练/评测链路，仍建议在确认驱动与 CUDA 栈兼容后再单独安装 `vllm`

## 5. 最小可验证复现路径

### 步骤 A：检查关键依赖是否可导入

```bash
python -c "import torch, transformers, datasets, sklearn, hydra, tensordict, wandb"
```

本次机器上的结果：

- 依赖导入已通过
- `torch` 版本为 `2.11.0+cu130`
- `transformers` 版本为 `4.57.6`
- `ray` 版本为 `2.54.1`
- `sglang` 版本为 `0.4.6`

但导入时出现一条重要告警：

- 当前 NVIDIA 驱动版本偏旧，不完全匹配 `torch 2.11.0+cu130`
- 这会影响后续 CUDA 推理或训练
- 如果要真正跑训练/评测，建议先确认驱动与 CUDA 版本兼容

### 步骤 B：处理评测数据

```bash
bash scripts/0_run_data_process.sh
```

这个脚本会：

- 下载 `RUC-NLPIR/FlashRAG_datasets`
- 调用 `taskutils/data_synthesis/process_test.py`
- 将处理后的测试数据写到 `data/test/`

预期结果：

- 生成 `data/test/eval_hotpotqa_*.json`
- 生成 `data/test/eval_2wikimultihopqa_*.json`

本次实际尝试结果：

- 该脚本在模块导入阶段就会执行 `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")`
- 当前机器无法稳定连通 Hugging Face
- 即使开启 `/dugaoyuan/clash/start.sh` 并设置 `HF_ENDPOINT=https://hf-mirror.com`，仍出现网络/SSL 错误
- 因此数据处理这一步目前卡在 tokenizer 下载，而不是 Python 依赖缺失

### 步骤 C：静态检查评测入口

评测脚本：

```bash
bash scripts/2_run_eval_ReMemR1.sh
```

但在真正运行前，必须先改 `taskutils/memory_eval/run_eval.py` 中的 checkpoint 路径：

- `ReMemR1_3B.ckpt="CHECKPOINT_PATH"`
- `ReMemR1_7B.ckpt="CHECKPOINT_PATH"`

如果直接评测公开模型，也应改成可用的 HuggingFace 模型名或本地合并后的 `hf_ckpt` 路径。

## 6. 官方结果复现路径

### 训练数据

按 README，需要将下面两个 parquet 文件放到 `data/train/`：

- `hotpotqa_train_32k.parquet`
- `hotpotqa_dev.parquet`

来源：

- `BytedTsinghua-SIA/hotpotqa`

### 训练

官方入口：

```bash
bash scripts/1_run_train_ReMemR1_3B.sh
bash scripts/1_run_train_ReMemR1_7B.sh
```

注意点：

- 默认脚本中 `N_NODE` 分别是 2 或更高，不适合当前单卡直接跑
- 训练脚本中的 `WANDB_API_KEY` 和 `WANDB_PROJECT` 需要替换
- 训练依赖 `ray + verl + sglang + FSDP`，资源要求高

如果只是本机试跑思路，建议先把：

- `N_NODE=1`
- `N_GPU=1`

并下调：

- `TRAIN_BS`
- `ROLLOUT_N`
- `ROLLOUT_VAL_N`
- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`

但这属于“调试跑通”，不等于官方结果复现。

### 合并 checkpoint

训练后先合并：

```bash
bash scripts/merge_ckpt.sh "results/memory_agent/ReMemR1_3B/global_step_200/actor"
```

输出目录通常是：

```bash
results/memory_agent/ReMemR1_3B/global_step_200/actor/hf_ckpt
```

### 评测

再执行：

```bash
bash scripts/2_run_eval_ReMemR1.sh
```

## 7. 当前建议的复现策略

对于当前这台机器，我建议按下面顺序推进：

1. 直接使用已创建好的 `rememr1_env`
2. 补齐缺失依赖并做导入测试
3. 跑 `scripts/0_run_data_process.sh`，确认评测数据生成
4. 将 `run_eval.py` 中 `CHECKPOINT_PATH` 改成真实模型路径
5. 如果只是验证推理链路，优先尝试评测公开 checkpoint
6. 如果要复现实验结果，再准备训练数据和多卡资源

## 8. 本次操作记录

本次排查已经确认：

- README 推荐 Python 3.11
- 已创建并验证新的 conda 环境 `rememr1_env`
- 项目没有独立环境文件，安装需参考 README
- 当前机器单卡可见，不满足官方多机多卡训练默认配置
- `run_eval.py` 中 ReMemR1 checkpoint 仍需手工填写
- 数据处理当前受阻于 Hugging Face tokenizer 下载
- 当前 `torch 2.11.0+cu130` 与本机驱动存在兼容性告警
- `doc/` 目录原本为空，现补充本说明文档

后续如果环境完成并且数据下载正常，建议继续把“依赖安装记录”和“实际运行日志摘要”追加到本文件下方。
