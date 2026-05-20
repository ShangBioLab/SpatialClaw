# ReMemR1 本机复现实录（2026-04-18）

## 1. 目标与范围

本文档记录的是在当前机器上对 `ReMemR1-main` 进行的一次**实际执行复现**，重点是：

- 确认仓库结构、环境和数据是否就绪
- 在本机单卡条件下打通最小可执行评测链路
- 记录过程中遇到的真实阻塞
- 给出最终成功跑通的命令、参数和结果文件位置

这份文档是对 `doc/reproduction_guide.md` 的补充。前者偏“规划与建议”，本文偏“实际执行记录”。

## 2. 机器与环境信息

本次实际确认到的环境如下：

- 项目目录：`/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main`
- GPU：`NVIDIA A800 80GB PCIe`
- 驱动：`525.105.17`
- 项目 Conda 环境：`rememr1_env`
- 独立模型服务环境：`vllm`

本次实际查到的关键 Python 包版本：

### `rememr1_env`

- `Python 3.11.15`
- `torch 2.11.0`
- `transformers 4.57.6`
- `ray 2.54.1`
- `sglang 0.4.6`
- `datasets 4.8.4`
- `wandb 0.26.0`

### `vllm`

- `Python 3.10.20`
- `vllm 0.18.0`
- `torch 2.10.0`
- `transformers 4.57.6`

## 3. 仓库就绪状态检查

本次执行时，仓库中以下资源已经存在：

- `data/train/hotpotqa_train_32k.parquet`
- `data/train/hotpotqa_dev.parquet`
- `data/test/eval_hotpotqa_*.json`
- `data/test/eval_2wikimultihopqa_*.json`
- `models/Qwen2.5-3B-Instruct/`

因此，这次复现没有再重新跑数据处理主流程，而是直接进入评测链路。

## 4. 实际遇到的阻塞

### 阻塞 1：本地模型分片未完整上传

最初 `models/Qwen2.5-3B-Instruct` 中的 safetensors 分片不完整，`vllm` 在载入时出现：

```text
SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered
```

后续逐个验证分片发现：

- `model-00001-of-00002.safetensors` 有问题
- `model-00002-of-00002.safetensors` 正常

在用户重新上传完整模型分片后，两份 safetensors 都可正常打开，问题消失。

### 阻塞 2：`sglang` 环境缺少 `pyzmq`

按仓库默认逻辑，本地评测优先可走 `sglang`。但在 `rememr1_env` 中启动服务时，出现：

```text
ModuleNotFoundError: No module named 'zmq'
```

这说明当前 `rememr1_env` 虽然装了 `sglang 0.4.6`，但没有配齐 `pyzmq` 依赖。

这次最终没有继续使用 `sglang` 路径作为主方案，而是切换到独立 `vllm` 环境起服务。

### 阻塞 3：`vllm 0.18.0` 默认编译路径失败

在 `vllm` 环境直接起服务时，模型能加载，但在 `torch.compile` 相关路径上失败，报错核心为：

```text
AttributeError: <function standalone_compile ...> does not have the attribute 'FakeTensorMode'
```

这不是模型权重问题，而是当前 `vllm 0.18.0` 与本机 Torch/编译路径组合下的兼容性问题。

解决方式是给 `vllm` 增加：

```bash
--enforce-eager
```

以关闭 `torch.compile` 和 `CUDAGraph` 路径。

### 阻塞 4：`vllm` 默认显存利用率略高

即使切换到 eager 模式，`vllm` 仍因为默认显存策略报错：

```text
ValueError: Free memory on device cuda:0 (...) is less than desired GPU memory utilization (0.9, ...)
```

解决方式是把显存占用目标从默认 `0.9` 下调到：

```bash
--gpu-memory-utilization 0.85
```

## 5. 为适配本机所做的代码改动

为了不给源码里硬编码服务参数，本次对 `taskutils/memory_eval/run_eval.py` 做了一个很小的兼容性修改：

- 新增环境变量 `EVAL_SERVER_ARGS`
- 在启动 `vllm` 或 `sglang` 服务命令时，将该变量追加到命令末尾

这样可以在不重复改源码的情况下，通过环境变量注入：

- `--enforce-eager`
- `--gpu-memory-utilization 0.85`

对应代码位置：

- `taskutils/memory_eval/run_eval.py` 中新增 `SERVER_ARGS = os.getenv("EVAL_SERVER_ARGS", "").strip()`
- 在 `serve()` 中用 `if SERVER_ARGS: cmd = f"{cmd} {SERVER_ARGS}"` 追加服务参数

## 6. 最终成功跑通的链路

本次最终采用的是：

- 评测主流程：`rememr1_env`
- 模型服务：`vllm` 环境
- 服务后端：`vllm`
- 评测模型：本地 `models/Qwen2.5-3B-Instruct`
- 评测方法：`rememr1`
- 任务：`eval_hotpotqa_50`
- GPU：单卡 `CUDA_VISIBLE_DEVICES=0`

说明：

- `eval_hotpotqa_50` 里的 `50` 表示每题上下文文档数为 50
- 实际载入的样本数为 `128`

## 7. 本次成功运行命令

本次成功使用的命令如下：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rememr1_env

export PROJECT_ROOT=/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main
export DATAROOT="$PROJECT_ROOT/data/test"
export REVERSED=0

export EVAL_SERVER_BACKEND=vllm
export EVAL_SERVER_PYTHON="/dugaoyuan/.conda/envs/vllm/bin/python"
export EVAL_SERVER_ARGS="--enforce-eager --gpu-memory-utilization 0.85"

export EVAL_DP=1
export EVAL_MODELS="ReMemR1-3B"
export EVAL_TASKS="eval_hotpotqa_50"

export REMEMR1_3B_CKPT="$PROJECT_ROOT/models/Qwen2.5-3B-Instruct"
export REMEMR1_3B_TOKENIZER="$PROJECT_ROOT/models/Qwen2.5-3B-Instruct"
export REMEMR1_3B_TP=1

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EXP_NAME="rememr1_3b_local_hotpotqa50_vllm_eager_gmu085"

bash scripts/2_run_eval_ReMemR1_local.sh
```

## 8. 实际运行过程摘要

这次成功运行的大致过程如下：

1. `run_eval.py` 启动本地服务命令
2. `vllm` 在独立环境中加载本地 `Qwen2.5-3B-Instruct`
3. 通过 `/v1/models` 探测确认服务已就绪
4. `taskutils/memory_eval/test_qa.py` 读取 `data/test/eval_hotpotqa_50.json`
5. 以 `api='rememr1'` 的方式并发调用本地 OpenAI 兼容接口
6. 完成全部 `128` 条样本推理
7. 写出 `jsonl` 预测结果和 `metric.json`

运行日志中可以看到：

- `Supported tasks: ['generate']`
- `Starting vLLM server on http://0.0.0.0:8000`
- 评测进度从 `0/128` 到 `128/128`

## 9. 最终结果

本次 `eval_hotpotqa_50` 的最终结果为：

```json
{
    "f1": 40.539062,
    "em": 22.65625,
    "sub_em": 39.0625
}
```

终端输出中对应为：

- `f1: 40.54`
- `em: 22.66`
- `sub_em: 39.06`
- `Total: 128`

## 10. 结果文件位置

本次结果已经写入：

- `taskutils/memory_eval/results/eval_hotpotqa_50/ReMemR1-3B.jsonl`
- `taskutils/memory_eval/results/eval_hotpotqa_50/ReMemR1-3B_metric.json`
- `taskutils/memory_eval/results/ReMemR1-3B.txt`

## 11. 对结果的正确理解

这次跑通的是“本机可执行复现链路”，但**不是论文最终结果复现**，原因如下：

- 当前使用的是本地 `Qwen2.5-3B-Instruct` 目录
- 不是训练后合并得到的正式 `ReMemR1` checkpoint
- 当前机器仍是单卡
- 官方训练脚本默认面向更高资源配置

因此，这次结果的意义主要是：

- 证明项目评测代码可以在本机单卡上打通
- 证明本地模型服务、数据读取、ReMemR1 推理逻辑和结果写出流程都能工作
- 为后续替换成真正的 `ReMemR1_3B` / `ReMemR1_7B` checkpoint 做准备

## 12. 后续建议

后续建议按下面顺序继续推进：

1. 准备真实的 `ReMemR1_3B` 或 `ReMemR1_7B` 合并后 checkpoint
2. 通过环境变量覆盖：
   - `REMEMR1_3B_CKPT`
   - `REMEMR1_3B_TOKENIZER`
3. 先跑：
   - `eval_hotpotqa_50`
   - `eval_2wikimultihopqa_50`
4. 再逐步扩展到更长文档档位：
   - `100`
   - `200`
   - `400`
   - `800`
   - `1600`
   - `3200`
   - `6400`

如果后续希望回到 `sglang` 路线，需要先把 `rememr1_env` 中的依赖补齐，至少确保：

- `pyzmq`
- `sglang`
- 与当前 CUDA/Torch 栈兼容的推理依赖

## 13. 本次复现的结论

结论可以概括为三点：

1. 仓库的本地评测链路在这台单卡 A800 机器上可以跑通
2. 成功路线不是 README 默认状态，而是：
   - `rememr1_env` 跑主流程
   - `vllm` 环境起服务
   - `--enforce-eager --gpu-memory-utilization 0.85`
3. 当前结果属于“本机评测链路复现成功”，不等同于“论文完整结果复现成功”
