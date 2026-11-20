# 设计：思维强度智能路由（Reasoning-Aware Routing）

> 状态：**已实现**（`app.py`）。本文件保留为设计说明与维护参考。
> 目标：在现有 Squilla Router 的「按难度选模型」之上，增加「按模型能力控制思维强度」。

---

## 1. 要解决什么

当前 Router 会算出 `thinking_mode`（T0–T3），但**只写进响应元数据，不下发给后端**
（`_call_backend` 构造 `outbound_body` 时，`thinking` / `effort` / `budget` 出现次数为 0）。

本设计把它变成真实生效的控制，并且：

- 只对**声明了支持的模型**下发（避免老模型 400）
- 各后端原生写法不同（OpenAI / Responses / Anthropic），由 llm-rosetta 自动转换
- 与 failover 正确配合（每个候选各自重算）
- 默认关闭时不改变任何现有行为

---

## 2. 核心决策

### 2.1 一次分类，两个维度

```
请求进来
   │
   ├─① 读取 agent 已表达的思维意图（若有）→ 归一化为 IR 三要素
   │
   ├─② 一次 ML 分类 → route_class(R0–R3) + thinking_mode(T0–T3)
   │
   ├─③ route_class → 选出候选 tier（模型）
   │
   ├─④ 对【每个候选】计算最终思维参数：
   │      合并 → clamp → 剥离 → 渲染成该后端原生写法
   │
   └─⑤ 发出；失败降级时，对**新候选**重新执行 ④
```

### 2.2 关键原则

| 原则 | 说明 |
|---|---|
| **意图可提前，参数必须后置** | agent 可以在选模型前表达"我想要高思维"，但真正的字段值必须等模型选定后按该模型能力生成 |
| **不支持就剥离** | 模型未声明思维能力时，整段思考参数不下发（不是尝试后再兜底） |
| **降级必须重算** | c3 渲染出的 `thinking.budget_tokens=4000` 绝不能带到 c1 |
| **默认零回归** | 全部配置不写 = 行为与当前完全一致 |

---

## 3. 统一表达：复用 llm-rosetta 的 IR 语汇

不自定义词汇，直接使用 llm-rosetta（已在依赖中）定义的类型：
`llm_rosetta.types.ir.reasoning.ReasoningConfig`

| 字段 | 取值 | 含义 |
|---|---|---|
| `mode` | `auto` / `enabled` / `disabled` | 是否进行显式思考 |
| `effort` | `minimal` / `low` / `medium` / `high` / `xhigh` / `max` | 强度阶梯 |
| `budget_tokens` | `int` | 绝对预算（可选） |
| `summary` | `auto` / `concise` / `detailed` / `none` | 思考内容可见性（可选） |
| `include_thoughts` | `bool` | Gemini 专用 |

### 各协议的原生落点（llm-rosetta 已实现，无需重写）

| IR | OpenAI Chat | OpenAI Responses | Anthropic |
|---|---|---|---|
| `effort: high` | `reasoning_effort: "high"` | `reasoning: {effort: "high"}` | `output_config: {effort: "high"}` |
| `mode: enabled` | — | `reasoning: {type: "enabled"}` | `thinking: {type: "enabled"}` |
| `mode: disabled` | `reasoning_effort: "none"` | `reasoning: {type: "disabled"}` | `thinking: {type: "disabled"}` |
| `mode: auto` | — | `reasoning: {type: "enabled"}`（模型自决） | `thinking: {type: "adaptive"}` |
| `budget_tokens` | — | — | `thinking: {budget_tokens: N}` |
| `summary` | `reasoning.summary` | `reasoning.summary` | 可见性映射 |

参考实现位置（已安装）：

- `llm_rosetta/converters/openai_chat/config_ops.py` — `ir_reasoning_config_to_p` / `p_reasoning_config_to_ir`
- `llm_rosetta/converters/openai_responses/config_ops.py` — 同上
- `llm_rosetta/converters/anthropic/config_ops.py` — 同上
- `llm_rosetta/converters/base/helpers/reasoning.py` — `apply_reasoning_config`（clamp / disabled 序列化）
- `llm_rosetta/shims/provider_shim.py` — `ReasoningCapability`（能力模型）
- `llm_rosetta/capabilities.py` — `enforce_reasoning`（能力兜底、剥离）

---

## 4. 配置文件设计（纯增量）

### 4.1 每个 tier 声明自己的能力

```env
# --- c0 = Qwen3.8-27B：支持 low/medium/xhigh，非连续能力 ---
C0_REASONING_MODES=auto,enabled,disabled
C0_REASONING_EFFORTS=low,medium,xhigh
C0_REASONING_DEFAULT_MODE=auto

# --- c1 = Qwen3.5-397B-A17B：完整 effort 区间 ---
C1_REASONING_MODES=auto,enabled,disabled
C1_REASONING_EFFORT_MIN=minimal
C1_REASONING_EFFORT_MAX=max
C1_REASONING_DEFAULT_MODE=auto

# --- c2 = GLM-5.3-Flash：完整 effort 区间 ---
C2_REASONING_MODES=auto,enabled,disabled
C2_REASONING_EFFORT_MIN=minimal
C2_REASONING_EFFORT_MAX=max
C2_REASONING_DEFAULT_MODE=auto

# --- c3 = DeepSeek-V4.1-Flash：完整 effort 区间 ---
C3_REASONING_MODES=auto,enabled,disabled
C3_REASONING_EFFORT_MIN=minimal
C3_REASONING_EFFORT_MAX=max
C3_REASONING_DEFAULT_MODE=auto

# 两个都写时，EFFORTS 白名单会再按 MIN/MAX 过滤。
```

**语义约定**

| 字段 | 缺省含义 |
|---|---|
| `REASONING_MODES` | 缺省 = **不支持**（保守，避免给老模型发未知字段） |
| `REASONING_EFFORT_MIN/MAX` | 缺省 = 使用该协议的默认区间 |
| `REASONING_EFFORTS` | 可选逗号白名单；用于 `low,medium,xhigh` 这类非连续能力 |
| `REASONING_DEFAULT_MODE` | 缺省 = `auto`（当模型支持 auto 时） |
| `REASONING_BUDGET_RATIO` | 缺省 = 不自动推导 budget |

> `disabled` 不在 `MODES` 列表里 → Router **不会**尝试关闭该模型的思考。

### 4.2 全局开关与映射

```env
ROUTER_REASONING_ENABLED=1        # 0 = 完全关闭（= 当前行为）
ROUTER_REASONING_PRIORITY=router  # router（自动分类优先）| agent（agent 显式指定优先）

THINKING_T0_EFFORT=minimal
THINKING_T1_EFFORT=low
THINKING_T2_EFFORT=medium
THINKING_T3_EFFORT=high
```

不写这几行时使用上面这套内置默认值。

### 4.3 热加载

现有 `_reload_config()`（`app.py`）会重新执行 `_discover_tiers()` 并重建
`TIERS` / `PROVIDERS` / `ROUTE_CLASS_TO_TIER`，因此**新增的 reasoning 字段自动跟随生效**，
无需额外实现。

---

## 5. 优先级与合并算法

### 5.1 优先级的两种模式

项目 `.env` 默认使用 `ROUTER_REASONING_PRIORITY=router`：始终采用分类器的
`T0-T3`，保证“模型档位 + 思维档位”来自同一次 Squilla 路由决策。

下表描述可选的 `ROUTER_REASONING_PRIORITY=agent` 模式：

| 情况 | 结果 |
|---|---|
| agent 带明确 `effort` | 用 agent 的值，再按候选模型 clamp |
| agent 带 `mode=enabled` + `budget_tokens` | 用 agent 的 budget（限制 < 该候选 `max_tokens`） |
| agent 带 `reasoning_effort: "none"` | 归一化为 `mode=disabled`；候选不允许则剥离 |
| agent 未带任何思维字段 | 用 Router 的 T0–T3 → effort 映射 |
| `ROUTER_REASONING_PRIORITY=router` | 忽略 agent 值，始终用 Router 推导 |

### 5.2 合并算法（伪代码）

```text
ir = normalize(agent_reasoning)            # rosetta: none -> disabled
if ir 为空 and ROUTER_REASONING_ENABLED:
    ir = {"effort": MAP[thinking_mode]}    # T0-T3 -> minimal..high
if not ROUTER_REASONING_ENABLED:
    ir = {}

cap = tier.reasoning_capability
if cap is None:                            # 模型未声明
    return {}                              # 整段剥离

# ① mode 过滤
if ir.mode and ir.mode not in cap.thinking_modes:
    if ir.mode == "disabled":
        return {}                          # 关不掉 -> 整段不发
    ir.mode = cap.thinking_default or "auto"

# ② effort clamp
if ir.effort:
    ir.effort = clamp(ir.effort, cap.effort_min, cap.effort_max)

# ③ budget 约束
if ir.budget_tokens:
    ir.budget_tokens = min(ir.budget_tokens, max_tokens - 1)
elif cap.budget_ratio:
    ir.budget_tokens = max(1024, int(max_tokens * cap.budget_ratio))
```

---

## 6. ✅ 实现方式：按 agent 方言注入（避免同格式短路）

`_formats_rosetta.convert_request()` 开头是：

```python
if inbound_fmt == outbound_fmt:
    return body
```

也就是说 **chat 进 → chat 后端** 时不会走 IR 渲染，注入的 IR 不会变成
`reasoning_effort`。

**实际采用的方案**：把最终生效的思维配置**直接写进 inbound（agent）方言**的请求体，
再交给 llm-rosetta 做跨格式转换。

- inbound 与 outbound 相同 → 注入的就是该方言的正确字段，无需再转换（短路无害）
- inbound 与 outbound 不同 → rosetta 解析该方言字段、渲染成后端方言

因此不需要 `force_conversion`。实现见 `app.py` 的 `_inject_reasoning()`。

**反向的坑（已修复）**：当某个候选**不支持思维**或**关不掉思考**时，必须把请求里
*已有的*思维字段一并剥离（包括 agent 自己传的），否则会原样转发导致后端 400。
实现见 `_strip_reasoning()`。

---

## 7. Failover 语义

每个候选 tier 的思维能力不同，因此：

1. **渲染必须在候选循环内部完成**，不能循环外算一次后复用
2. 降级到**不支持思维**的 tier → **必须整段剥离**，否则 400
3. 降级到**范围更窄**的 tier → **必须重新 clamp**

对应修改位置：`app.py` 的 `_call_backend()` 内，
`outbound_body = dict(body); outbound_body["model"] = cand_backend["model"]`
之后、`convert_request(...)` 之前。

---

## 8. 可观测性

### 8.1 响应元数据

在 `_router` 中增加：

```json
"_router": {
  "tier": "c1",
  "model": "GLM-5.3-Flash",
  "thinking_mode": "T2",
  "reasoning": {
    "source": "router",
    "requested": "T2",
    "effective": {"mode": "auto", "effort": "medium"},
    "clamped": false,
    "dropped_reason": null
  }
}
```

`source` 取值：`agent` / `router` / `none`
`dropped_reason` 取值：`unsupported_model` / `disabled_not_allowed` / `global_off`

### 8.2 日志

```text
[REASONING] tier=c1 model=GLM-5.3-Flash src=router T2 -> effort=medium clamped=false
```

### 8.3 `/router-status`

输出各 tier 的 reasoning 能力，便于排查"为什么没生效"。

---

## 9. 边界情况

| 情况 | 处理 |
|---|---|
| Anthropic `budget_tokens` ≥ `max_tokens` | clamp 到 `max_tokens - 1` |
| Anthropic thinking 与 `temperature` 冲突 | 启用 thinking 时丢弃 temperature（或用协议默认） |
| `max_tokens` 很小（如 100） | 不做 budget 自动推导，只设 effort |
| 模型"永远思考"（deepseek-flash） | 不配置 `disabled`，Router 不会尝试关闭 |
| chat 进 → anthropic 后端 | rosetta 渲染为 `thinking` / `output_config.effort` |
| tools + thinking 同时出现 | 两者可共存，`tool_calls` 流式处理不受影响 |
| 流式返回 reasoning 增量 | 现有代码已识别 `reasoning_content`，无需改动 |
| reasoning 占满 `max_tokens` | 现有逻辑按截断透传（不降级），启用思维控制后可显著减少 |

---

## 10. 实现清单

| # | 文件 | 位置 | 改动 |
|---|---|---|---|
| 1 | `app.py` | `_discover_tiers()` | 解析 `C<N>_REASONING_*` 字段 |
| 2 | `app.py` | 新增函数 | `_merge_reasoning()` / `_clamp_effort()` / T0–T3 映射 |
| 3 | `_formats_rosetta.py` | `convert_request()` | 支持 `force_conversion` 透传 |
| 4 | `app.py` | 三个端点的 `_classify` 之后 | 把 `thinking_mode` 传入 `_call_backend` |
| 5 | `app.py` | `_call_backend()` 候选循环内 | 按候选注入 reasoning + 转换 |
| 6 | `app.py` | `_call_backend()` 的 `_router` | 增加 `reasoning` 块 |
| 7 | `app.py` | `/router-status` | 输出各 tier reasoning 能力 |
| 8 | `.env.example` / `README.md` | — | 文档化新字段 |
| 9 | `_reload_config()` | 已覆盖 | 无需改动 |

**不需要改动**：

- `final_policy.py`（T0–T3 已经算好，可直接复用）
- `_cd_sse.py` / `_cd_translate.py`（流式与消息翻译已兼容 reasoning）
- `_formats.py`（已弃用）

---

## 11. 测试计划

| # | 场景 | 预期 |
|---|---|---|
| 1 | 所有 tier 不配 reasoning | 行为与当前完全一致（0 回归） |
| 2 | chat 进 → chat 后端 + T3 | 后端收到 `reasoning_effort: "high"` |
| 3 | chat 进 → anthropic 后端 + T2 | 后端收到 `thinking.type=adaptive` + `output_config.effort=medium` |
| 4 | anthropic 进带 `budget_tokens` → chat 后端 | 转成 effort，budget 被丢弃 |
| 5 | agent 带 `reasoning_effort=none` → 不可关闭模型 | 剥离字段，请求仍 200 |
| 6 | 模型 `EFFORT_MAX=high` + T3（映射 max） | clamp 成 high，`_router.reasoning.clamped=true` |
| 7 | c3 降级到 c1（范围更窄） | 用 c1 的 cap 重算，不复用 c3 字段 |
| 8 | 降级到不支持思维的 tier | 字段剥离，不 400 |
| 9 | `ROUTER_REASONING_ENABLED=0` | 回到当前行为 |
| 10 | 改 `.env` 增删 reasoning 字段 | 热加载生效，无需重启 |

---

## 12. 分阶段实施建议

| 阶段 | 内容 | 验证点 |
|---|---|---|
| **一** | 配置项 + 合并算法 + `force_conversion` | 测试 1、2、9（确认零回归） |
| **二** | 注入 + 逐候选重算 | 测试 3、6、7、8 |
| **三** | 可观测性 + 文档 | 测试 4、5、10 |

每个阶段可独立验证，避免一次性引入过多不确定性。

---

## 13. 与多模态能力的对比

| 维度 | 多模态（已实现） | 思维强度（本设计） |
|---|---|---|
| 能力声明 | `.env` 手写 `SUPPORTS_IMAGE/VIDEO` | `.env` 手写 `REASONING_*` |
| 自动发现 | 不支持（主流 MaaS 不暴露元数据） | 不支持（同理） |
| 路由维度 | 先筛能力，再按 T0–T3 分类 | 直接复用同一个 `thinking_mode` |
| 协议差异 | 由各 provider 原样处理 | 由 llm-rosetta 自动转换 |
| 默认行为 | 不声明=不支持 | 不声明=不支持 |
| 热加载 | 支持 | 支持（复用同一机制） |
