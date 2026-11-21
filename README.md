# Squilla API Router

一个单进程、本地运行的模型路由网关。它接收 OpenAI Chat、OpenAI Responses 或 Anthropic Messages 请求，用本地 SquillaRouter V4 分类器选择模型等级，并把请求按目标 Provider 的原生 API 格式转发。

本项目只做模型路由，不修改用户提示词或工具定义。限流、配额、并发和请求大小由后端 MaaS 负责。

## 核心能力

- OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 三种入口
- 请求、响应和 SSE 流式格式自动转换
- 本地 ML 分类与 c0..cN 分级路由
- 多 Provider、多 Tier 配置
- Provider 故障 fallback、共享熔断状态
- Session sticky、任务变化判断、软重路由
- 图片/视频能力筛选
- 思维强度映射与协议转换
- 按大小轮转的应用日志和 access log

## 快速启动

```bash
cd /path/to/squilla_api_router
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env
python _run_server.py --host 127.0.0.1 --port 8021
```

任务边界模型需要以下文件：

```text
model_bundle/task_change/v1.joblib
model_bundle/task_change/v1.joblib.sha256
model_bundle/bge_onnx/
```

校验模型：

```bash
cd model_bundle/task_change
sha256sum -c v1.joblib.sha256
```

启动后检查：

```bash
curl -s http://127.0.0.1:8021/health
curl -s http://127.0.0.1:8021/router-status
```

## API 格式

| 端点 | 客户端格式 |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages |

客户端使用哪种格式，响应就返回哪种格式。无论目标后端原生使用哪种格式，Router 都会自动转换。

## Provider 匹配

推荐使用显式 Provider 命名空间：

```env
PROVIDER_<NAME>_BASE_URL
PROVIDER_<NAME>_API_KEY
PROVIDER_<NAME>_FORMAT
```

`<NAME>` 会标准化为小写，Tier 通过 `C<N>_PROVIDER=<NAME>` 精确引用。

示例：

```env
PROVIDER_PRIMARY_BASE_URL=https://primary.example/v1
PROVIDER_PRIMARY_API_KEY=replace-me
PROVIDER_PRIMARY_FORMAT=openai_chat

PROVIDER_SECONDARY_BASE_URL=https://secondary.example/v1
PROVIDER_SECONDARY_API_KEY=replace-me
PROVIDER_SECONDARY_FORMAT=openai_responses
```

兼容旧式 `<PREFIX>_BASE_URL`，此时 Provider 名就是 `<PREFIX>` 小写。若两种写法定义同一名称，`PROVIDER_*` 显式配置优先。不存在代码写死的 Provider 名称别名。

`FORMAT` 可用值：

```text
openai_chat
openai_responses
anthropic
```

未填写时，路径含 `/anthropic` 或 `/messages` 推断为 Anthropic，其他默认 OpenAI Chat。

## Tier 配置

任何 `C<N>_MODEL` 会生成一个 Tier。数量不限。

```env
C0_PROVIDER=primary
C0_MODEL=your-model-c0
C0_SUPPORTS_IMAGE=1
C0_SUPPORTS_VIDEO=0

C1_PROVIDER=secondary
C1_MODEL=your-model-c1
C1_SUPPORTS_IMAGE=1
C1_SUPPORTS_VIDEO=0

C2_PROVIDER=primary
C2_MODEL=your-model-c2
C2_SUPPORTS_IMAGE=1
C2_SUPPORTS_VIDEO=1

C3_PROVIDER=secondary
C3_MODEL=your-model-c3
C3_SUPPORTS_IMAGE=1
C3_SUPPORTS_VIDEO=0
```

可选能力：

```env
# 未配置表示上下文上限未知，不做上下文能力过滤
C0_CONTEXT_LENGTH=131072

# 思维强度能力
C0_REASONING_MODES=auto,enabled,disabled
C0_REASONING_EFFORTS=low,medium,high
C0_REASONING_DEFAULT_MODE=auto
```

## 最小示例配置

```env
ROUTER_API_KEY=replace-with-a-long-random-key

PROVIDER_PRIMARY_BASE_URL=https://primary.example/v1
PROVIDER_PRIMARY_API_KEY=replace-me
PROVIDER_PRIMARY_FORMAT=openai_chat

C0_PROVIDER=primary
C0_MODEL=your-model-c0
C0_SUPPORTS_IMAGE=0
C0_SUPPORTS_VIDEO=0

ROUTER_SESSION_STICKY=1
ROUTER_TASK_CHANGE_MODE=safe_hybrid
ROUTER_TASK_CHANGE_HARD_THRESHOLD=0.45
ROUTER_TASK_CHANGE_SOFT_THRESHOLD=0.20
ROUTER_TASK_CHANGE_MODEL=model_bundle/task_change/v1.joblib

ROUTER_LOG_ENABLED=1
ROUTER_LOG_LEVEL=INFO
ROUTER_LOG_FILE=squilla_router.log
ROUTER_ACCESS_LOG_FILE=squilla_router_access.log
```

## 鉴权

```env
ROUTER_API_KEY=replace-with-a-long-random-key
```

支持：

```text
Authorization: Bearer <key>
x-api-key: <key>
```

可选租户映射：

```env
ROUTER_TENANT_API_KEYS={"team-a":"router-key-a","team-b":"router-key-b"}
```

租户 Key 只能访问模型 API，不能访问 `/admin/*`。全局 `ROUTER_API_KEY` 是管理员 Key。

## Session 与任务变化

请求可携带：

```text
X-Squilla-Session-Id
X-Squilla-Task-Id
X-Squilla-Intent: new_task|continue|redo|task_changed
X-Squilla-Quality-Feedback
```

`ROUTER_TASK_CHANGE_MODE=safe_hybrid` 采用 fail-safe 策略：

```text
p(change) >= hard_threshold
  -> 创建新 task 并重新选择模型

soft_threshold <= p(change) < hard_threshold
  -> 保持 task_id，但本轮强制重新分类并更新滚动任务目标

p(change) < soft_threshold
  -> 允许 sticky

模型或文件不可用
  -> 退回规则；模糊轮次仍强制重新分类
```

相关响应头：

```text
X-Squilla-Force-Reclassify
X-Squilla-Task-Boundary
X-Squilla-Task-Change-Probability
```

重新训练：

```bash
python train_task_change_model.py
python benchmarks/benchmark_task_change_hybrid.py --repeat 3
```

## 思维强度路由

```env
ROUTER_REASONING_ENABLED=1
ROUTER_REASONING_PRIORITY=router
```

`PRIORITY=router` 表示自动分类优先；`PRIORITY=agent` 表示优先采用 Agent 显式传入的 thinking/reasoning 参数，再按模型能力 clamp。

模型不支持思维控制时不要配置 `C<N>_REASONING_*`。fallback 到其他 Tier 时会按新模型能力重新计算。

## 多模态路由

图片和视频请求先按 Tier 能力筛选，再在可用 Tier 内运行文本难度分类。

```env
C0_SUPPORTS_IMAGE=1
C0_SUPPORTS_VIDEO=0
```

只有支持对应模态的 Tier 才会成为候选。没有可用 Tier 时返回：

```json
{"error":{"code":"NO_CAPABLE_BACKEND"}}
```

旧 `VISION_*` 兜底配置已废弃。

## Failover 与超时

```env
BACKEND_TIMEOUT=600
BACKEND_CONNECT_TIMEOUT=30
BACKEND_STREAM_START_TIMEOUT=120
BACKEND_STREAM_FAST_START_TIMEOUT=8
BACKEND_STREAM_IDLE_TIMEOUT=600
```

- 非流式使用 `BACKEND_TIMEOUT`
- 流式首个有效内容之前使用 start timeout
- T0/T1 可使用更短的 fast start timeout
- 内容开始后使用 idle timeout
- HTTP 4xx/5xx、连接失败、无首内容等会触发候选链 fallback

熔断配置：

```env
ROUTER_CIRCUIT_BREAKER_ENABLED=1
ROUTER_FAILURE_COOLDOWN_SECONDS=60
ROUTER_RATE_LIMIT_COOLDOWN_SECONDS=15
```

熔断状态保存在 SQLite，可由多个进程共享。

## 日志

```env
ROUTER_LOG_ENABLED=1
ROUTER_LOG_LEVEL=INFO
ROUTER_LOG_FILE=squilla_router.log
ROUTER_ACCESS_LOG_FILE=squilla_router_access.log
ROUTER_LOG_MAX_BYTES=20971520
ROUTER_LOG_BACKUP_COUNT=5
```

排查路由时使用：

```env
ROUTER_LOG_LEVEL=DEBUG
```

重点日志：

```text
[TASK_BOUNDARY]   任务边界、软重分类和概率
[MODEL_SWITCH]    计划切换 Tier/model
[ATTEMPT]         实际后端尝试
[STREAM]          首内容超时、HTTP 错误、fallback
[CIRCUIT]         熔断跳过
[REASONING]       最终发送的思维强度字段
```

access log 只记录最终 HTTP 状态，不能单独区分 timeout、上游 4xx/5xx、连接失败或 fallback 耗尽。

### 502 排查

```bash
grep -E \
'TASK_BOUNDARY|MODEL_SWITCH|ATTEMPT|STREAM|CIRCUIT|all backends failed|HTTP [45][0-9][0-9]|first content' \
squilla_router.log | tail -n 300
```

常见原因：

- `no first content within Ns`
- 上游返回 4xx/5xx/429
- 连接重置或超时
- 所有 fallback 候选失败
- Redo 的质量下限使候选只剩单一 Tier

## 管理端点

| 端点 | 说明 |
|---|---|
| `GET /health` | 健康检查 |
| `GET /router-status` | Provider、Tier、路由和模型状态 |
| `POST /admin/reload-config` | 热加载 `.env` |
| `GET /admin/routing-state` | 查询 Session 状态 |
| `DELETE /admin/routing-state` | 删除 Session 状态 |
| `POST /admin/purge-state` | 清理过期状态 |
| `GET /admin/circuit-breakers` | 查询熔断状态 |
| `GET /admin/routing-usage` | 查询 usage 和失败账本 |

除健康检查和状态查询外，管理端点都需要管理员 Key。

## OpenCode 示例

```json
{
  "provider": {
    "router": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8021/v1",
        "apiKey": "your-router-key"
      }
    }
  }
}
```

## 生产检查清单

- 所有 Router/Provider Key 已使用独立的强随机密钥
- `.env` 不提交到 Git
- `task_change_policy_ready=true`
- Provider 和 Tier 均出现在 `/router-status`
- 日志、状态库和模型路径使用预期目录
- MaaS 后端模型名与 Tier 配置完全一致
- 目标 MaaS 已验证 Chat、Responses 或 Anthropic 原生格式
- 流式首内容超时符合实际推理模型行为
- 已对 fallback、熔断和 Session sticky 做真实多轮验证
- pickle/joblib 模型与生产 Python、scikit-learn、LightGBM 版本一致

## 实测结果

以下数据来自仓库内的真实 E2E 输出，不使用理论估算。

### OpenCode 多轮工具调用 A/B

同一 Agent 任务分别使用 Router 和固定最高 Tier：

| 指标 | Router | 固定高 Tier | 变化 |
|---|---:|---:|---:|
| Agent steps | 3 | 4 | -1 |
| Tool calls | 3 | 4 | -1 |
| Total tokens | 28,584 | 37,364 | **-23.50%** |
| Cost units | 299,160 | 2,434,360 | **-87.71%** |
| Elapsed time | 25.32s | 23.65s | +7.06% |

来源：

```text
benchmarks/opencode_ab_summary.json
```

该结果同时减少 tokens 和费用，但单次任务存在输出长度、工具轮次和缓存状态差异，
不能外推为所有请求的固定收益。

### 12 任务 E2E

同一批任务比较 Router 和固定高 Tier：

| 指标 | Router | 固定高 Tier | 变化 |
|---|---:|---:|---:|
| Total tokens | 14,858 | 5,605 | +165.08% |
| Cost units | 3,251,160 | 3,888,800 | **-16.40%** |
| Total latency | 214.97s | 202.08s | +6.38% |

来源：

```text
benchmarks/e2e_p1_vs_classifier.json
```

这组结果说明：Router 的价值主要是选择更低价格或更合适的模型，并不保证 token
数量一定减少。部分低成本模型可能输出更长，因此必须同时比较 tokens 和价格。

### 结果解释

- `cost_units` 使用配置中的 Tier 价格倍率和真实 usage 计算。
- 不同 Provider 的实际价格、缓存计费、输出长度和失败重试会影响最终结果。
- 上线前应使用自己的模型价格和真实 Agent 请求重新运行基准。
- 对短回答、长代码、工具调用和长上下文应分别统计，不能只看单一平均值。

## 测试

```bash
python -m pytest tests -q
python benchmarks/benchmark_task_change.py --repeat 3 --sweep
python benchmarks/benchmark_task_change_hybrid.py --repeat 3
```

## 依赖

见 `requirements.txt`。生产环境应固定实际使用的 Python 和依赖版本，尤其是 `scikit-learn`、`lightgbm`、`onnxruntime` 和 `joblib`。

## 许可

- 本项目代码：Apache 2.0
- BGE 模型：以其上游许可为准
- SquillaRouter 分类器来源：OpenSquilla 及其上游项目
