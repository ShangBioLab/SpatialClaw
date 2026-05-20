# ReMemR1 本机复现实录（2026-04-22）

## 1. 本次目标

本次工作的目标是继续在当前机器上复现 `ReMemR1-main`，并确认：

- 仓库里此前为本机适配的评测链路是否仍然可用
- `rememr1_env + vllm` 的分环境方案是否还能稳定复跑
- 在不覆盖 2026-04-18 历史结果的前提下，重新产出一份新的评测结果

这次仍然是“评测链路复现”，不是训练复现。

## 2. 本次实际确认到的环境

- 项目目录：`/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main`
- GPU：`NVIDIA A800 80GB PCIe`
- 驱动：`525.105.17`
- 评测主环境：`rememr1_env`
- 模型服务环境：`vllm`

本次再次确认到的关键环境状态：

- `rememr1_env`
  - `Python 3.11.15`
  - `torch 2.11.0+cu130`
  - `ray 2.54.1`
  - `pyzmq` 仍未安装
- `vllm`
  - `Python 3.10.20`
  - `torch 2.10.0+cu128`
  - `vllm 0.18.0`

## 3. 仓库就绪状态

本次执行前，仓库中这些资源已经存在且可用：

- 本地模型：`models/Qwen2.5-3B-Instruct/`
- 评测数据：`data/test/eval_hotpotqa_50.json`
- 历史评测结果：
  - `taskutils/memory_eval/results/eval_hotpotqa_50/ReMemR1-3B.jsonl`
  - `taskutils/memory_eval/results/eval_hotpotqa_50/ReMemR1-3B_metric.json`

说明：

- 数据处理这一步本次没有重新执行
- 本次直接复跑最小单卡评测链路

## 4. 本次实际检查到的代码状态

`taskutils/memory_eval/run_eval.py` 中，之前为本机评测加入的能力仍然存在：

- 可通过 `EVAL_SERVER_BACKEND` 切换 `vllm` / `sglang`
- 可通过 `EVAL_SERVER_PYTHON` 指定独立服务环境解释器
- 可通过 `EVAL_SERVER_ARGS` 追加服务启动参数

这说明仓库中的“本机单卡评测方案”已经不是临时命令，而是已有代码基础支持。

## 5. 本次新增的小改动

为了安全复跑而不覆盖旧结果，本次又对 `taskutils/memory_eval/run_eval.py` 做了两处小改动：

### 5.1 新增结果文件控制

新增环境变量：

- `EVAL_SAVE_SUFFIX`
- `EVAL_FORCE`

作用：

- 可以给结果文件加后缀，例如 `ReMemR1-3B-repro-20260422`
- 可以通过环境变量开启 `--force`
- 这样复跑不会覆盖 2026-04-18 的基线结果

### 5.2 修正清理阶段的假异常

成功评测结束后，原来的 `Config.__del__` 会再次尝试杀已退出的服务进程组，导致：

```text
ProcessLookupError: [Errno 3] No such process
```

本次增加了一个 `safe_kill_process_group()`，只在进程仍存活时才发信号，从而避免这个收尾噪音。

## 6. 本次第一次尝试的真实失败点

第一次仍按 2026-04-18 的参数启动：

```bash
--enforce-eager --gpu-memory-utilization 0.85
```

这次失败，核心原因不是代码，而是当前 GPU 可用显存比 4 月 18 日更紧。`vllm` 实际报错为：

```text
ValueError: Free memory on device cuda:0 (...) is less than desired GPU memory utilization (0.85, ...)
```

因此，本次没有沿用 `0.85`，而是改为更保守的：

```bash
--gpu-memory-utilization 0.83
```

## 7. 本次最终成功运行命令

本次成功复跑使用的命令如下：

```bash
export PATH=/dugaoyuan/.conda/envs/rememr1_env/bin:$PATH
export PROJECT_ROOT=/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main
cd "$PROJECT_ROOT"

export DATAROOT="$PROJECT_ROOT/data/test"
export REVERSED=0

export EVAL_SERVER_BACKEND=vllm
export EVAL_SERVER_PYTHON="/dugaoyuan/.conda/envs/vllm/bin/python"
export EVAL_SERVER_ARGS="--enforce-eager --gpu-memory-utilization 0.83"

export EVAL_DP=1
export EVAL_MODELS="ReMemR1-3B"
export EVAL_TASKS="eval_hotpotqa_50"

export REMEMR1_3B_CKPT="$PROJECT_ROOT/models/Qwen2.5-3B-Instruct"
export REMEMR1_3B_TOKENIZER="$PROJECT_ROOT/models/Qwen2.5-3B-Instruct"
export REMEMR1_3B_TP=1

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export EVAL_FORCE=1
export EVAL_SAVE_SUFFIX="-repro-20260422"
export EXP_NAME="rememr1_3b_local_hotpotqa50_vllm_eager_gmu083_repro_20260422"

bash scripts/2_run_eval_ReMemR1_local.sh
```

## 8. 本次实际运行过程摘要

本次运行的大致过程如下：

1. `run_eval.py` 读取环境变量并启动本地 `vllm` 服务
2. `vllm` 在独立环境中加载本地 `Qwen2.5-3B-Instruct`
3. 通过 `/v1/models` 探针确认服务就绪
4. `taskutils/memory_eval/test_qa.py` 读取 `eval_hotpotqa_50.json`
5. 使用 `api='rememr1'` 并发调用本地 OpenAI 兼容接口
6. 完成全部 `128` 条样本推理
7. 写出新结果文件

## 9. 本次复现结果

本次 `eval_hotpotqa_50` 的最终结果为：

```json
{
    "f1": 39.078125,
    "em": 21.09375,
    "sub_em": 40.625
}
```

终端输出对应为：

- `f1: 39.08`
- `em: 21.09`
- `sub_em: 40.62`
- `Total: 128`

## 10. 与 2026-04-18 结果的对比

2026-04-18 的历史结果为：

```json
{
    "f1": 40.539062,
    "em": 22.65625,
    "sub_em": 39.0625
}
```

本次结果略有波动，但链路本身是成功的。考虑到本次并不是训练后正式 checkpoint，而是对本地 `Qwen2.5-3B-Instruct` 做 ReMemR1 推理式评测，这种波动是可以接受的。

## 11. 本次新结果文件位置

本次新结果已写入：

- `taskutils/memory_eval/results/eval_hotpotqa_50/ReMemR1-3B-repro-20260422.jsonl`
- `taskutils/memory_eval/results/eval_hotpotqa_50/ReMemR1-3B-repro-20260422_metric.json`
- `taskutils/memory_eval/results/ReMemR1-3B-repro-20260422.txt`

旧结果文件未被覆盖。

## 12. 本次复现能说明什么

本次复现说明：

- 当前仓库中的本机单卡评测链路仍然可用
- `rememr1_env` 和 `vllm` 分环境方案仍然成立
- 本地模型、数据、结果写出逻辑都可正常工作
- 当前机器上更稳妥的 `vllm` 设置应为：
  - `--enforce-eager`
  - `--gpu-memory-utilization 0.83`

## 13. 仍未解决的问题

### 13.1 这仍然不是训练复现

本次只验证了评测链路，不代表：

- RL 训练已打通
- checkpoint 合并链路已验证
- 论文正式结果已复现

### 13.2 `sglang` 路径仍未恢复

当前 `rememr1_env` 里仍缺：

```bash
pyzmq
```

如果后续要回到 README 默认的 `sglang` 训练/评测路径，还需要补这个依赖。

### 13.3 单卡训练仍未跑通

仓库中的单卡 smoke 训练日志显示，当前机器在训练链路里首先遇到的是 Ray/dashboard agent 异常，而不是模型前向本身的错误。

## 14. 下一步建议

建议按下面顺序推进：

1. 如果目标是继续“评测复现”，直接沿用本次命令模板，扩大到更多测试集
2. 如果目标是“训练复现”，先看 `doc/training_guide_20260422.md`
3. 真正训练后，按 README 执行：
   - `bash scripts/merge_ckpt.sh "<actor_dir>"`
4. 再把合并后的 `hf_ckpt` 用到评测中

