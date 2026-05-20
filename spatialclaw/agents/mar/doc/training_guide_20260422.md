# ReMemR1 训练说明（基于 README 与当前机器状态，2026-04-22）

## 1. 先明确：我们之前做的是评测，不是训练

此前已经打通的是：

- 本地模型服务启动
- `taskutils/memory_eval` 评测流程
- ReMemR1 风格的推理与结果写出

但这不等于训练过了 ReMemR1。真正训练 ReMemR1 的链路应该是：

1. 准备训练数据
2. 启动 PPO/RL 训练
3. 训练结束后合并 checkpoint
4. 再拿合并后的 checkpoint 去评测

README 里对应的训练入口是：

```bash
bash scripts/1_run_train_ReMemR1_3B.sh
bash scripts/1_run_train_ReMemR1_7B.sh
```

## 2. 训练前需要准备什么

### 2.1 环境

README 推荐的基础安装方式是：

```bash
conda create -n rememr1 python=3.11
conda activate rememr1
pip install httpx==0.23.1 aiohttp -U ray[serve,default] vllm
pip install nltk pyyaml beautifulsoup4 html2text wonderwords tenacity fire
pip install vllm==0.9 --index-url https://download.pytorch.org/whl/cu126
pip install "sglang==0.4.6"
pip install hydra-core accelerate tensordict torchdata wandb "tensordict<=0.6.2"
```

在当前机器上，更现实的做法是直接使用已经存在的：

```bash
rememr1_env
```

但要注意两点：

- 训练命令必须在 `rememr1_env` 里跑，不要误跑到 `vllm` 环境
- 如果使用 README 默认的 `sglang` rollout，当前环境还应补装：

```bash
pip install pyzmq
```

### 2.2 数据

训练脚本依赖以下两个文件：

- `data/train/hotpotqa_train_32k.parquet`
- `data/train/hotpotqa_dev.parquet`

仓库当前已经有这两个文件，所以这一步目前是满足的。

### 2.3 基座模型

训练脚本默认基座模型为：

- 3B：`models/Qwen2.5-3B-Instruct` 或 `Qwen/Qwen2.5-3B-Instruct`
- 7B：`Qwen/Qwen2.5-7B-Instruct`

当前仓库已存在本地 3B 模型目录，因此 3B 训练更适合先做本机调试。

### 2.4 日志与 WandB

训练脚本要求：

- `WANDB_API_KEY`
- `WANDB_PROJECT`

如果只是本机调试，建议先离线：

```bash
export WANDB_MODE=offline
```

## 3. README 所说的“正式训练”是什么意思

README 的默认设计是面向多机多卡 RL 训练：

- 3B 脚本已被你当前仓库改成更偏本机可调试版本
- 7B 脚本仍然保持明显的多机多卡设定

例如：

- `scripts/1_run_train_ReMemR1_7B.sh` 默认：
  - `N_NODE=4`
  - `N_GPU=8`
  - `actor_rollout_ref.rollout.name=sglang`
  - `actor_rollout_ref.actor.fsdp_config.fsdp_size=8`

这和“当前单卡 A800”不是同一个资源级别。

所以你需要区分两种目标：

### 目标 A：官方资源条件下的训练复现

这意味着：

- 多机多卡
- Ray 集群
- 更接近 README 默认脚本
- 更有希望逼近论文结果

### 目标 B：当前机器上的最小训练打通

这意味着：

- `N_NODE=1`
- `N_GPU=1`
- 大幅下调 batch、rollout、token 长度
- 重点是让链路“能跑起来”
- 不要把这种结果理解成论文级训练复现

## 4. 在当前机器上，建议先做 3B 单卡调试训练

原因很简单：

- 3B 本地模型已经在
- 单卡 A800 更容易先把 3B 打通
- 7B 即便能跑，也更容易在 rollout、Ray、显存利用上遇到额外问题

## 5. 推荐的训练步骤

### 步骤 1：进入正确环境

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rememr1_env
cd /dugaoyuan/AgentAndClaw/Agents/ReMemR1-main
```

建议补一项：

```bash
pip install pyzmq
```

如果后面仍不用 `sglang`，这一步不是绝对必须，但建议补齐。

### 步骤 2：先做一个最小 smoke run

当前仓库里的 `scripts/1_run_train_ReMemR1_3B.sh` 已经被改造成适合本机试跑的版本，支持通过环境变量覆盖主要训练参数。

建议先用非常保守的配置：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rememr1_env
cd /dugaoyuan/AgentAndClaw/Agents/ReMemR1-main

export WANDB_MODE=offline
export WANDB_PROJECT=rememr1_local_debug
export WANDB_API_KEY=dummy

export CUDA_VISIBLE_DEVICES=0

export EXP_LOG_NAME=ReMemR1_3B_single_gpu_smoke
export MODEL_PATH=/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main/models/Qwen2.5-3B-Instruct

export N_NODE=1
export N_GPU=1
export FSDP_SIZE=1

export MAXLEN=2048
export MAX_NEW_TOKEN=256
export TRAIN_BS=4
export PPO_MINI_BS=1
export TOTAL_EPOCHS=1
export SAVE_FREQ=1
export TEST_FREQ=1

export ROLLOUT_N=2
export ROLLOUT_VAL_N=1
export VAL_BEFORE_TRAIN=false

export ROLLOUT_BACKEND=vllm
export ROLLOUT_TP=1
export ROLLOUT_GPU_UTIL=0.45
export ROLLOUT_ENFORCE_EAGER=true
export ROLLOUT_FREE_CACHE_ENGINE=false
export ROLLOUT_MAX_BATCHED_TOKENS=4096

export ACTOR_MAX_TOKENS_PER_GPU=4096
export REF_MAX_TOKENS_PER_GPU=4096
export ROLLOUT_LOGPROB_MAX_TOKENS_PER_GPU=4096

bash scripts/1_run_train_ReMemR1_3B.sh
```

这组参数的目的不是训出好模型，而是尽量降低资源压力，验证：

- Hydra 配置能否正确展开
- Ray 是否能稳定拉起
- actor / rollout / ref 是否都能初始化
- 数据读入是否正常

## 6. 当前机器上，训练真正卡在哪里

截至 2026-04-22，这台机器上的训练阻塞已经分成两个阶段：

### 6.1 第一阶段阻塞：Ray 本地实例启动后，dashboard agent 提前退出

最初单卡 smoke 训练确实首先死在 Ray：

```text
raylet exited immediately because one Ray agent failed, agent_name = dashboard_agent
```

这一阶段现在已经被基本绕开，做法包括：

- 显式使用 `rememr1_env` 的 `python3` 和 `wandb`
- 关闭 `RAY_enable_pipe_based_agent_to_parent_health_check`
- 在本机训练脚本里补充：
  - `RAY_FORCE_MINIMAL_DASHBOARD_AGENT=1`
  - `RAY_SKIP_DASHBOARD_AGENT_MODULES=1`

在这些修复后，Ray 本地集群已经可以稳定启动，`TaskRunner.run` 能进入真实训练准备阶段。

### 6.2 第二阶段阻塞：当前训练环境的 Torch/CUDA 与驱动不兼容

当 smoke 训练进一步进入 worker 初始化后，新的硬阻塞出现于：

- `verl/single_controller/base/worker.py`
- `Worker.__init__()`
- 调用 `torch.cuda.get_device_name()`

实际报错为：

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 12000).
```

这不是 ReMemR1 代码本身的问题，而是：

- 当前 `rememr1_env` 使用：
  - `torch 2.11.0+cu130`
- 当前机器驱动为：
  - `525.105.17`
  - 对应 `CUDA 12.0`

因此，Ray 问题解决后，训练仍然会卡在 GPU 初始化阶段。

## 7. 当前实际 smoke training 已经推进到哪一步

截至本次实际执行，单卡 3B smoke training 已经能走到：

1. `main_ppo` 正常启动
2. 本地 Ray 正常拉起
3. `TaskRunner.run` 正常执行
4. 训练/验证 parquet 正常读取
5. prompt filtering 正常完成
6. dataloader 正常构建
7. 进入 `trainer.init_workers()`

随后在 worker 创建阶段失败，失败点是：

```text
torch.cuda.get_device_name() -> torch._C._cuda_init()
```

说明目前“训练链路起不来”的主要剩余问题已经不是 Ray，而是 GPU 驱动与 Torch 版本匹配。

## 8. 现在真正需要做什么

如果要继续训练，当前最现实的方案有两个：

### 方案 A：升级系统驱动

这是最直接、最干净的方案。

目标是让当前机器驱动能够兼容：

```bash
torch 2.11.0+cu130
```

如果驱动升级完成，当前 `rememr1_env` 更有希望直接继续训练。

### 方案 B：换成更低版本的 Torch 训练环境

由于这台机器上：

- `vllm` 环境中的 `torch 2.10.0+cu128`

已经实际跑通过推理，因此理论上可以考虑：

1. 以 `vllm` 环境为基底
2. 补齐训练所缺依赖
3. 再尝试运行训练

但当前 `vllm` 环境还缺一些训练依赖，例如：

- `accelerate`

并且它的 Ray 版本也与当前训练调试环境不一致，所以这条路仍然需要额外整理，不如直接升级驱动来得稳妥。

## 9. 如果训练成功，结果会产出到哪里

以 3B 脚本为例，训练结果默认会落到：

```bash
results/memory_agent/<EXP_LOG_NAME>/
```

例如：

```bash
results/memory_agent/ReMemR1_3B_single_gpu_smoke/
```

训练日志通常会在：

```bash
log/<EXP_LOG_NAME>.log
```

## 10. 训练后如何合并 checkpoint

README 里要求训练后先合并 checkpoint，然后再评测。

当前仓库提供了：

```bash
bash scripts/merge_ckpt.sh "<actor_dir>"
```

例如：

```bash
bash scripts/merge_ckpt.sh "results/memory_agent/ReMemR1_3B/global_step_200/actor"
```

这个脚本会：

1. 调用 `scripts/model_merger.py`
2. 从 `huggingface/` 目录合并权重
3. 输出到：

```bash
<actor_dir>/hf_ckpt
```

## 11. 合并后如何评测

合并好以后，把评测 checkpoint 指向新的 `hf_ckpt` 即可。

例如：

```bash
export REMEMR1_3B_CKPT="/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main/results/memory_agent/ReMemR1_3B/global_step_200/actor/hf_ckpt"
export REMEMR1_3B_TOKENIZER="/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main/results/memory_agent/ReMemR1_3B/global_step_200/actor/hf_ckpt"
```

然后再跑：

```bash
bash scripts/2_run_eval_ReMemR1_local.sh
```

## 12. 你现在最合理的推进顺序

如果目标是“把训练链路真正推进下去”，建议按这个顺序：

1. 保留当前已经修好的 Ray workaround
2. 优先解决驱动与 Torch 版本不兼容问题
3. 补 `pyzmq`
4. 再跑 3B 单卡 smoke
5. smoke 通过后，再逐步增大：
   - `MAXLEN`
   - `ROLLOUT_N`
   - `TRAIN_BS`
   - `TOTAL_EPOCHS`
6. 训练成功后执行 `merge_ckpt.sh`
7. 再切到评测脚本验证效果

## 13. 一句话总结

README 里的“训练 ReMemR1”不是直接跑评测脚本，而是：

```bash
准备 data/train -> 跑 scripts/1_run_train_ReMemR1_3B.sh 或 7B.sh -> 合并 checkpoint -> 再评测
```

在你当前这台机器上，单卡 `3B` smoke 训练已经推进到真正的 worker 初始化阶段；当前最关键的剩余阻塞已经不是 Ray，而是 `rememr1_env` 中 `torch 2.11.0+cu130` 与本机 NVIDIA 驱动版本不兼容。

## 14. 2026-04-23 更新：`vllm` 路线训练推进结果

在 `rememr1_env` 被 `torch 2.11.0+cu130` 与本机驱动不兼容卡住后，已经切到第 2 条路：以 `vllm` 环境为训练基底继续推进。

截至 2026-04-23，这条路线已经确认：

- `vllm` 环境可正常做 CUDA 张量计算，能绕开 `rememr1_env` 的驱动/Torch 阻塞。
- 通过本地 Ray dashboard-agent workaround，训练不再死在 Ray 启动早期。
- 通过关闭 `use_remove_padding`，已绕开 repo 对 `flash_attn.bert_padding` 的强依赖。
- 通过把 `accelerate` 处理成当前路径下的可选依赖，已绕开 `accelerate` 缺失。
- 通过新增 `MODEL_ATTN_IMPL` 并切到 `sdpa`，已绕开 Transformers 在加载 Qwen2.5-3B 时对 `FlashAttention2` 的硬依赖。

最新 smoke run 为：

- `ReMemR1_3B_single_gpu_smoke_20260422_vllm5`
- 日志：`log/ReMemR1_3B_single_gpu_smoke_20260422_vllm5.log`

它已经推进到：

- `trainer.init_workers()`
- `WorkerDict` 实际加载 `Qwen2.5-3B-Instruct` checkpoint shards

当前最新硬阻塞变成：

- `WorkerDict` 在 checkpoint 加载完成后、`torch.distributed.barrier()` / NCCL 同步附近触发 `SIGSEGV`
- 不再是缺包问题，而是更底层的 FSDP/NCCL runtime 稳定性问题

更完整的本轮过程、命令和结论见：

- `doc/training_progress_20260423_vllm_route2.md`
