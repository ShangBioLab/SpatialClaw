# ReMemR1 训练推进记录：Route 2 (`vllm` 环境转训练环境)

日期：2026-04-23

## 1. 背景

在 `rememr1_env` 中，单卡 smoke 训练已经推进到真实 worker 初始化，但最终被本机驱动与 `torch 2.11.0+cu130` 的兼容性卡住：

- GPU: `NVIDIA A800 80GB PCIe`
- Driver: `525.105.17`
- `rememr1_env` Torch: `2.11.0+cu130`
- 典型报错：`The NVIDIA driver on your system is too old (found version 12000)`

因此本轮改走第 2 条路：以 `vllm` 环境为训练基底继续推进。

## 2. `vllm` 环境实际情况

已确认：

- Python: `3.10.20`
- Torch: `2.10.0+cu128`
- Ray: `2.44.1`
- CUDA 张量与 `torch.cuda.get_device_name(0)` 可正常工作
- `verl.trainer.main_ppo` 可以直接 import
- 缺失依赖：`accelerate`
- 不存在本地 `flash_attn`

## 3. 本轮做过的代码与环境调整

### 3.1 Ray 本地 workaround 迁移到 `vllm` 环境

为了避开 Ray dashboard agent 在本机异常退出的问题，已把和 `rememr1_env` 类似的 workaround 迁移到：

- `/dugaoyuan/.conda/envs/vllm/lib/python3.10/site-packages/ray/dashboard/agent.py`
- `/dugaoyuan/.conda/envs/vllm/lib/python3.10/site-packages/ray/_private/services.py`

效果：

- `ray.init(local_mode=True)` 在 `vllm` 环境中可稳定成功
- 训练不再在 Ray 启动早期直接因 dashboard agent 崩溃退出

### 3.2 训练脚本支持关闭 remove padding

已调整：

- `scripts/1_run_train_ReMemR1_3B.sh`

新增环境变量：

- `MODEL_USE_REMOVE_PADDING`

用途：

- 当前机器没有 `flash_attn.bert_padding`
- 可以通过 `MODEL_USE_REMOVE_PADDING=false` 让 actor/ref/critic 走非 rmpad 路径

### 3.3 actor / critic 侧将 `flash_attn.bert_padding` 变为按需依赖

已调整：

- `verl/workers/actor/dp_actor.py`
- `verl/workers/critic/dp_critic.py`

效果：

- 不再在模块 import 阶段因为缺 `flash_attn` 直接失败
- 只有 `use_remove_padding=True` 时才会明确要求 `flash_attn`

### 3.4 `accelerate` 改为当前路径下的可选依赖

已调整：

- `verl/utils/fsdp_utils.py`
- `verl/utils/model.py`
- `verl/utils/checkpoint/megatron_checkpoint_manager.py`

效果：

- 单卡源 rank 路径不再因为 `from accelerate import init_empty_weights` 无条件导入而失败
- 非源 rank 或 Megatron 空权重初始化路径仍会在真正需要时给出明确报错

### 3.5 attention backend 改为可配置

已调整：

- `verl/workers/fsdp_workers.py`
- `scripts/1_run_train_ReMemR1_3B.sh`

新增环境变量：

- `MODEL_ATTN_IMPL`

用途：

- 原代码在多个 `from_pretrained(...)` 处硬编码 `attn_implementation="flash_attention_2"`
- 当前机器没有 `flash_attn`，因此显式切到 `sdpa` 才能继续推进

## 4. 本轮 smoke 训练结果

### 4.1 `vllm1`

结果：失败

结论：

- CUDA 正常
- 训练进入 `main_ppo`
- 但在真正训练过程中依然触发 Ray dashboard agent 崩溃

### 4.2 `vllm2`

结果：失败

结论：

- 迁移 Ray workaround 后，已穿过原先的 dashboard-agent 早退问题
- 到达 `trainer.init_workers()` 相关阶段
- 首个新阻塞是：`ModuleNotFoundError: No module named 'flash_attn'`

日志：

- `/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main/log/ReMemR1_3B_single_gpu_smoke_20260422_vllm2.log`

### 4.3 `vllm3`

参数关键点：

- `MODEL_USE_REMOVE_PADDING=false`

结果：失败

结论：

- 成功绕过 repo 自己对 `flash_attn.bert_padding` 的强依赖
- 新阻塞变成：`ModuleNotFoundError: No module named 'accelerate'`

日志：

- `/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main/log/ReMemR1_3B_single_gpu_smoke_20260422_vllm3.log`

### 4.4 `vllm4`

参数关键点：

- `MODEL_USE_REMOVE_PADDING=false`
- `accelerate` 已改为当前路径下可选依赖

结果：失败

结论：

- 成功绕过 `accelerate` 缺失
- 新阻塞变成 Hugging Face / Transformers 在加载 Qwen2.5-3B 时强制尝试 `FlashAttention2`
- 典型报错：`FlashAttention2 has been toggled on ... the package flash_attn seems to be not installed`

日志：

- `/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main/log/ReMemR1_3B_single_gpu_smoke_20260422_vllm4.log`

### 4.5 `vllm5`

参数关键点：

- `MODEL_USE_REMOVE_PADDING=false`
- `MODEL_ATTN_IMPL=sdpa`

结果：失败，但再次明显前进

已经确认通过的阶段：

- Ray 本地启动
- Hydra 配置展开
- 数据过滤
- `trainer.init_workers()` 开始执行
- `WorkerDict` 开始真实加载 `Qwen2.5-3B-Instruct` checkpoint shards
- `FlashAttention2` 缺失问题已被 `sdpa` 绕开

最终新阻塞：

- `WorkerDict` 在加载完 checkpoint shards 后，进入 `torch.distributed.barrier()` / NCCL 相关阶段附近触发 `SIGSEGV`
- 关键日志：
  - `Loading checkpoint shards: 100%|...| 2/2`
  - `ProcessGroupNCCL.cpp:5138 Guessing device ID based on global rank`
  - `*** SIGSEGV received ...`
  - `ActorDiedError`

日志：

- `/dugaoyuan/AgentAndClaw/Agents/ReMemR1-main/log/ReMemR1_3B_single_gpu_smoke_20260422_vllm5.log`

## 5. 到 2026-04-23 为止的真实结论

`vllm` 路线已经证明：

1. 这条路线可以绕开 `rememr1_env` 的驱动/Torch 不兼容问题。
2. Ray 本地启动问题可以通过 dashboard-agent workaround 暂时绕过。
3. `flash_attn` 和 `accelerate` 在当前单卡 smoke 路径里并非不可逾越，已经可以通过代码和参数继续往前推进。
4. 当前最新硬阻塞已经收敛为：
   - 单卡 `WorkerDict` 在 `sdpa + FSDP + NCCL barrier` 附近发生原生段错误（`SIGSEGV`）
   - 这比之前的环境缺包问题更底层，也更接近真正的训练 runtime 问题

## 6. 当前最推荐的下一步

按优先级建议：

1. 优先缩小 `vllm5` 的 NCCL/FSDP 段错误范围。
   - 重点排查 `verl/workers/fsdp_workers.py` 里 model load 完成后的 `torch.distributed.barrier()` 与 FSDP 初始化顺序。
   - 重点看单卡场景是否存在不必要的 NCCL barrier / process group 初始化。
2. 如果只是为了先把训练链路“单卡打通”，可以尝试：
   - 单卡场景下减少或跳过某些分布式同步步骤
   - 明确给 `init_process_group` 传 `device_id`
   - 继续下调可能触发不稳定行为的 FSDP / offload 设置
3. 如果目标改回“尽量接近官方训练设置”，则仍然建议：
   - 直接准备一个具备 `flash_attn` 的标准训练环境
   - 或升级驱动后回到更标准的训练栈

## 7. 本轮最小可复用命令模板

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
cd /dugaoyuan/AgentAndClaw/Agents/ReMemR1-main

export WANDB_MODE=offline
export WANDB_PROJECT=rememr1_local_debug
export WANDB_API_KEY=dummy
export CUDA_VISIBLE_DEVICES=0

export PYTHON_BIN=/dugaoyuan/.conda/envs/vllm/bin/python3
export WANDB_BIN=/dugaoyuan/.conda/envs/vllm/bin/wandb

export EXP_LOG_NAME=ReMemR1_3B_single_gpu_smoke_20260422_vllm5
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

export MODEL_USE_REMOVE_PADDING=false
export MODEL_ATTN_IMPL=sdpa

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

## 8. 一句话总结

到 2026-04-23 为止，`vllm` 路线已经把 ReMemR1 的单卡训练 smoke 从“环境无法进入训练”推进到了“3B checkpoint 可实际加载，但在 NCCL/FSDP 同步附近出现段错误”。这说明当前最大的剩余问题已经不再是依赖缺失，而是更底层的分布式训练 runtime 稳定性。
