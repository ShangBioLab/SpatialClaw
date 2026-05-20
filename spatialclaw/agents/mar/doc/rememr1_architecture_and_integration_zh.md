# ReMemR1 代码架构与集成指南

## 1. 项目定位

ReMemR1 的核心目标，不是单纯把超长上下文一次性塞给模型，而是把长文档处理改造成一个可训练的循环过程：

1. 把长文档切成 chunk。
2. 模型每轮只看一个 chunk。
3. 模型维护一份逐步更新的 memory。
4. 当发现当前 chunk 的信息不够时，模型可以通过 `<recall>query</recall>` 触发“回看”历史 memory。
5. 全部 chunk 处理完后，再基于累计的 memory 输出最终答案。

这套设计把“长上下文理解”拆成了“顺序阅读 + 可回访记忆 + 最终作答”三层，因此它更像一个带外部记忆策略的 agent，而不是普通单轮 QA 模型。

## 2. 仓库结构总览

这个仓库可以分成四个核心层次。

| 目录 | 作用 | 关键文件 |
| --- | --- | --- |
| `recurrent/` | ReMemR1 的 agent 抽象、循环执行框架、记忆实现 | `interface.py`、`generation_manager.py`、`utils.py`、`impls/memory_revisit.py` |
| `verl/` | 训练基础设施，负责 rollout、GRPO/PPO、reward、分布式训练 | `trainer/ppo/ray_trainer.py`、`trainer/ppo/metric_utils.py` |
| `taskutils/memory_eval/` | 独立评测脚本与推理包装 | `run_eval.py`、`utils/rememr1.py` |
| `scripts/` | 数据处理、训练、合并 ckpt、评测启动脚本 | `0_run_data_process.sh`、`1_run_train_ReMemR1_7B.sh`、`2_run_eval_ReMemR1.sh` |

如果只看“能不能集成进你自己的 agent 系统”，最关键的是前两层：

- `recurrent/` 决定 agent 怎么读 chunk、怎么更新 memory、怎么 recall。
- `verl/` 决定这套行为怎么被训练出来。

## 3. 最核心的抽象接口

### 3.1 `recurrent/interface.py`

这里定义了整个 recurrent agent 插件机制。

- `RConfig`
  - 每种 recurrent agent 的配置基类。
  - 具体实现通过 dataclass 扩展自己的字段。

- `RDataset`
  - recurrent 版本的数据集基类，继承自 `RLHFDataset`。
  - 重点不是普通的 `input_ids`，而是把长上下文拆成 agent 能消费的字段，比如 `context_ids`、`context_length`、`prompt_ids`。

- `RAgent`
  - 同步 recurrent agent 的生命周期接口：
  - `start()` 初始化状态
  - `action()` 生成当前轮 prompt
  - `update()` 消化 LLM 输出并更新内部状态
  - `done()` 决定循环是否结束
  - `end()` 输出 `final_mask` 和 `sample_index`

- `AsyncRAgent`
  - 异步版本接口，面向单 sample async rollout。
  - 更适合直接对接 chat completion server。

- `RRegister`
  - 插件注册器。
  - 仓库通过 `RRegister.from_filename(path, name)` 动态加载具体实现。
  - 这意味着 ReMemR1 的 memory agent 不是硬编码到 trainer 里的，而是一个可替换模块。

这部分的意义很大：如果你自己的 agent 系统已经有 planner / tool / memory 框架，那么你不一定要照搬 `verl`，但可以直接借鉴这套 `RAgent` 生命周期抽象。

## 4. ReMemR1 主实现思路

### 4.1 `recurrent/impls/memory_revisit.py`

这是当前仓库里最重要、也最值得参考的实现。

它主要包含三个对象：

- `MemoryConfig`
- `MemoryDataset`
- `MemoryAgent`

### 4.2 `MemoryDataset`

`MemoryDataset` 会把单条样本整理成几类信息：

- `context_ids`
  - 长文档的 token 序列。
- `context_length`
  - 有效上下文长度。
- `prompt_ids`
  - 问题部分。

其中一个关键点是：它把 `data.max_prompt_length` 改写成 `max_chunks * chunk_size`，本质上是在数据层面约束“最多读多少 chunk”。

### 4.3 `MemoryAgent` 的状态机

`MemoryAgent` 在 `start()` 里初始化几类状态：

- `self.memory`
  - 当前轮可继续传递给后续 chunk 的 memory。
- `self.history_memory`
  - 历史 memory 集合，用于后续 recall 检索。
- `self.recall_memories`
  - 最近一次被召回的历史 memory。
- `self.step`
  - 当前读到第几个 chunk。

之后每轮循环做三件事：

1. `action()`
   - 判断哪些 sample 还没读完。
   - 从 `context_ids` 中切出当前 chunk。
   - 将 `prompt + recalled_memory + memory + chunk` 拼成当前轮输入。
   - 如果 chunk 已经读完，则切换到 final 模板，只保留 `prompt + recalled_memory + memory`，要求输出 `\boxed{}` 最终答案。

2. `generate`
   - 真正调用 rollout worker 做一次 LLM 生成。
   - 这部分不在 `MemoryAgent` 内部，而是交给 `LLMGenerationManager`。

3. `update()`
   - 从模型输出里解析 `<recall>...</recall>`。
   - 用 TF-IDF 从 `history_memory` 里找 top1 历史 memory。
   - 提取新的 memory 文本并覆盖当前 `self.memory`。
   - 把新 memory 写入 `history_memory`，供后续 recall 使用。

这就是 ReMemR1 的核心闭环：

`chunk_t -> memory_t -> recall(history_1...t) -> memory_t+1 -> final answer`

### 4.4 Prompt 设计

主模板里有四块动态内容：

- `<problem>`
- `<recalled_memory>`
- `<memory>`
- `<section>`

模型被要求在一次输出里给出：

- `<thinking>...</thinking>`
- `<update>...</update>`
- 可选 `<recall>query</recall>`

最终轮模板则要求输出带 `\boxed{}` 的答案。

这说明 ReMemR1 不是把 recall 做成真实 tool call API，而是把 recall 先表示成一个语言动作，再由外部控制器解析并执行。

这点对集成很关键：你完全可以把 `<recall>` 替换成你自己 agent 系统里的函数调用、memory search API 或 event。

## 5. 执行控制层

### 5.1 `recurrent/generation_manager.py`

`LLMGenerationManager` 是同步执行的总控。

它负责：

- 调用 `agent.start()`
- 循环执行 `agent.action() -> rollout -> agent.update()`
- 收集每一步的 `gen_output`
- 在结束后拼成完整 trajectory
- 生成 `final_mask` 和 `sample_index`

这里有两个非常重要的工程点。

### 5.2 `sample_index` 和 `final_mask`

由于一个原始问题会扩展成多步 trajectory，所以训练时必须维护“当前 step 属于哪个原始 sample”。

- `sample_index`
  - 记录每一步对应原始 batch 中哪个样本。

- `final_mask`
  - 标记哪些 step 是最终答案步。

这两个张量后面会直接参与：

- final answer reward 对齐
- GRPO advantage 对齐
- 只取 final step 做结果评分

如果你要把这套逻辑接进自己的 agent 训练框架，这两个字段建议保留。它们本质上就是“多步轨迹回映射”机制。

### 5.3 `TokenTemplate`

`recurrent/utils.py` 里的 `TokenTemplate` 很值得保留。

它不是字符串 format 后再 tokenize，而是：

1. 预先把模板静态片段 tokenize。
2. 动态字段直接拼 token tensor。

这样做的好处：

- 避免重复 decode/encode。
- 大 batch、多轮循环下更稳。
- 对长上下文训练更省时间。

如果你自己的 agent 系统也会高频做“模板 + memory + chunk”的拼接，建议保留这个思路。

## 6. Recall / Memory 检索机制

### 6.1 `recurrent/impls/tf_idf_retriever.py`

当前仓库没有上向量库，也没有用额外 encoder，而是用一个很轻的 TF-IDF 检索器。

流程很简单：

1. 把 `history_memory` 当成 corpus。
2. 用 tokenizer 分词。
3. 对 `<recall>query</recall>` 做 TF-IDF 相似度检索。
4. 返回 top1 memory。

它的优点是：

- 非常轻。
- 没有额外服务依赖。
- 对“从自己历史摘要里找回一句相关信息”这类任务已经够用。

它的缺点也很明显：

- 只能做浅层 lexical matching。
- memory 一旦写得很抽象，TF-IDF 会退化。
- 历史 memory 多了以后，质量和速度都不是最优。

所以如果你要集成到自己的 agent 系统：

- 快速落地阶段：TF-IDF 足够。
- 生产级系统：建议换成 embedding 检索或你现有的 memory store。

## 7. 训练框架如何接入 recurrent agent

### 7.1 `verl/trainer/ppo/ray_trainer.py`

ReMemR1 对 `verl` 的改造，核心是三处切换：

1. 配置检查阶段
   - 如果 `config.recurrent.enable` 开启，就通过 `RRegister.from_filename()` 动态加载 recurrent 实现。

2. 数据集阶段
   - `train_dataset` 和 `val_dataset` 改为 `recurrent_register.dataset_cls(...)`。

3. rollout 阶段
   - 不再直接生成一次 response。
   - 改为通过 `generation_manager.run_llm_loop_revisit(...)` 执行完整的多步记忆循环。

也就是说，trainer 眼里看到的不是单步生成，而是一条被展开后的多步轨迹。

### 7.2 为什么它能在 RL 里工作

因为它把 agent 的多步行为变成了一种“可展开的 sequence trajectory”：

- 每一步都有 prompt / response。
- 每一步都能挂 reward。
- 每一步都能映射回原始样本。
- 最终还能只对 final step 做 outcome reward。

这是它最值得迁移的设计，不一定非要和 `verl` 绑定。

## 8. Reward 设计思路

### 8.1 `verl/trainer/ppo/metric_utils.py`

ReMemR1 的 reward 不是只看最终答对没，而是分两层。

#### 第一层：Outcome Reward

- 来自 final answer。
- 通过 `compute_reward(...)` 调用任务 reward，比如 HotpotQA 的 EM/F1。
- 只对 final step 真正有监督意义。

#### 第二层：State Reward

对每一步动作都打分，主要包括：

- `compute_format_rewards`
  - 检查输出格式是否正确。
  - memory step 是否恰好包含一个 `<update>`。
  - final step 是否恰好包含一个 `\boxed{}`。

- `compute_action_rewards`
  - 检查 memory 更新是否让 ground truth 相关信息覆盖更高。
  - 检查 recalled memory 是否给当前 prompt 带来额外有效信息。

最终 trainer 用：

`alpha * outcome_adv + (1 - alpha) * state_adv`

把两层优势混合起来。

### 8.2 这套 reward 的真正价值

它解决的是一个 RL 里很常见的问题：

- 如果只奖励最终答对，模型学不会中间过程该怎么更新 memory。
- 如果只奖励中间动作，模型会学成“格式很好但结果没用”。

ReMemR1 的做法是同时约束：

- 终局正确性
- 中间状态质量

这也是你把它迁到自己 agent 系统时最该保留的部分之一。

## 9. 评测链路

### 9.1 `taskutils/memory_eval/utils/rememr1.py`

评测侧并没有直接复用 trainer，而是做了一个轻量推理包装：

1. 读取 context 和 prompt。
2. 按 `RECURRENT_CHUNK_SIZE` 切 chunk。
3. 每轮调用 `/chat/completions`。
4. 解析 `<recall>` 和 memory 更新结果。
5. 全部读完后发 final prompt，拿最终答案。

也就是说，ReMemR1 在“推理部署”层面其实非常容易移植，因为它最终只依赖一个 OpenAI 风格的 chat completion 接口。

### 9.2 一个需要注意的点

仓库里的 async 代码有旧实现痕迹：

- `recurrent/impls/async_memory.py` 引用了并不存在的 `recurrent.impls.memory`
- 其模板参数也和 `memory_revisit.py` 现版本不完全一致

所以如果你要真正复用当前仓库主线，建议以 `memory_revisit.py` 和 `taskutils/memory_eval/utils/rememr1.py` 这两条同步逻辑为准，不要直接把 `async_memory.py` 当成权威实现。

## 10. 如何集成进你自己的 agent 系统

下面分两种路线。

### 10.1 路线 A：只集成推理时的“可回访记忆”

这是最推荐的落地方式，成本最低。

#### 你需要引入的最小模块

1. `MemoryController`
   - 保存 `current_memory`
   - 保存 `history_memory`
   - 负责 recall 检索

2. `ChunkScheduler`
   - 把长文档切成 chunk
   - 控制 step 递增

3. `PromptAssembler`
   - 负责拼装：
   - `problem`
   - `recalled_memory`
   - `memory`
   - `section`

4. `OutputParser`
   - 从模型输出里提取：
   - `<update>`
   - `<recall>`
   - final answer

#### 最小执行伪代码

```python
history_memory = []
current_memory = "No previous memory"
recalled_memory = "No memory was recalled."

for chunk in chunk_document(context, chunk_size):
    prompt = build_memory_prompt(
        question=question,
        chunk=chunk,
        memory=current_memory,
        recalled_memory=recalled_memory,
    )
    response = llm(prompt)
    new_memory = parse_update(response)
    recall_query = parse_recall(response)

    if new_memory:
        current_memory = new_memory
        history_memory.append(new_memory)

    if recall_query:
        recalled_memory = memory_search(recall_query, history_memory)
    else:
        recalled_memory = "No memory was recalled."

final_prompt = build_final_prompt(
    question=question,
    memory=current_memory,
    recalled_memory=recalled_memory,
)
final_answer = llm(final_prompt)
```

#### 适合集成到哪些系统

- 你已经有 agent runtime，只缺长文档记忆机制。
- 你有自己的 tool / planner，不想引入 `verl`。
- 你只想要 inference enhancement，不想做 RL 训练。

### 10.2 路线 B：把 ReMemR1 作为“可训练的 memory policy”接进你现有训练框架

如果你自己的 agent 系统已经有 RL、偏好优化或 trajectory-level training，那么可以进一步把 ReMemR1 的训练思想接进去。

#### 你至少需要保留的结构

1. 数据层
   - 每条样本要拆成：
   - `question`
   - `context_ids`
   - `context_length`
   - ground truth / reward target

2. 轨迹层
   - 每个 sample 会展开成多步 action trajectory。
   - 必须保留 `sample_index` 和 `final_mask`。

3. reward 层
   - final outcome reward
   - per-step format reward
   - per-step memory/action reward

4. advantage 层
   - 能对“同一问题的多个 rollout”做 group-based advantage。

如果缺少第 2 点和第 4 点，你很难真正复现 ReMemR1 的训练方式。

### 10.3 如果你的 agent 系统已经有 memory 模块，应该怎么嫁接

最稳的做法不是替换你现有 memory，而是给现有系统增加一个“ReMemR1 风格的 reading policy”。

建议拆成三层：

1. `MemoryWritePolicy`
   - 决定当前 chunk 读完后，memory 怎么更新。

2. `MemoryRecallPolicy`
   - 决定什么时候 recall，以及 recall 用什么 query。

3. `MemoryStore`
   - 真正存历史 memory。
   - 可替换为：
   - in-memory list
   - Redis / KV
   - 向量库
   - 你自己的 episodic memory 系统

ReMemR1 当前仓库里把这三者耦合在 `MemoryAgent.update()` 中了。你自己系统里最好把它们拆开，这样更易维护。

### 10.4 推荐的工程化改造

如果你要把它集成进正式系统，我建议做下面几件事。

#### 改造 1：把 memory 从“自由文本”改成结构化对象

当前实现里，memory 本质是模型输出的一段文本。这样快，但不稳定。

建议改成：

```json
{
  "facts": [],
  "entities": [],
  "relations": [],
  "uncertainties": [],
  "citations": []
}
```

好处：

- recall 检索更稳定
- 可做去重
- 更容易和 tool / planner 共享

#### 改造 2：把 recall 从 XML 标签升级成真实函数调用

当前实现：

```text
<recall>who is ...</recall>
```

更实用的集成方式：

```json
{"tool": "memory_search", "query": "...", "top_k": 3}
```

这样可以直接接你的 tool-calling runtime。

#### 改造 3：把 TF-IDF 换成你已有的 retrieval backend

推荐优先级：

1. 如果是快速验证，保留 TF-IDF。
2. 如果你已有 embedding memory store，直接接入。
3. 如果你有 knowledge graph / entity store，可把 recall query 分流到结构化检索。

#### 改造 4：保留 final synthesis 独立一步

不要把“读最后一个 chunk”与“输出最终答案”合并。ReMemR1 把 final answer 独立成一轮是正确的，因为：

- 可以明确区分 memory update 和 answer generation。
- final reward 计算更干净。
- 便于后续做 self-consistency 或 best-of-n。

## 11. 我建议你的接入顺序

### 阶段 1：先做 inference-only 版本

目标：

- 不动你现有 agent 大框架。
- 只在“长文档任务”上引入 chunked memory + recall。

你要做的只有：

1. 接入 chunk 切分器。
2. 接入 memory 状态。
3. 接入 recall 检索。
4. 保留 final synthesis。

### 阶段 2：再做 reward instrumentation

即便你暂时不做 RL，也建议先把这些日志打出来：

- 每轮 `<update>` 是否存在
- recall query 是什么
- recall 命中到了哪条 memory
- memory 长度变化
- final answer 正确率

这样后面要做训练或行为分析时，代价会低很多。

### 阶段 3：最后再考虑 RL 化

只有在下面三个前提都成立时，才建议上 ReMemR1 式训练：

1. 你的 agent 已经稳定运行 inference 版本。
2. 你有可批量计算的最终任务 reward。
3. 你能承担多步 trajectory 展开的训练成本。

否则先做 inference 增强，ROI 更高。

## 12. 这个仓库里最值得复用的代码

如果你只想高效迁移，不建议整仓硬搬，建议优先复用这些点：

- `recurrent/interface.py`
  - 生命周期抽象和插件注册方式。

- `recurrent/impls/memory_revisit.py`
  - MemoryAgent 主逻辑。

- `recurrent/generation_manager.py`
  - 多步轨迹控制。

- `recurrent/utils.py`
  - `TokenTemplate`、padding、`final_batch`。

- `taskutils/memory_eval/utils/rememr1.py`
  - 最容易改造成你自己线上推理流程的参考版本。

如果你还要训练，再补：

- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/metric_utils.py`

## 13. 最后给你的落地建议

如果你的目标是“把 ReMemR1 的能力集成到自己的 agent 系统”，建议不要把它理解成“再加一个 memory 模块”，而要理解成：

“给 agent 增加一个可训练的、可回访的长文档阅读策略。”

最小可用迁移路径是：

1. 保留 chunk-by-chunk 阅读。
2. 保留 `memory / history_memory / recalled_memory` 三态。
3. 保留 final synthesis 独立一步。
4. 先用你自己的 memory 检索实现替换 TF-IDF。
5. 等 inference 版本稳定后，再决定是否引入 RL reward。

如果你按这个顺序做，改动面最小，收益也最大。
