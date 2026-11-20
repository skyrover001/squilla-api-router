"""Build expanded training data for the offline task-change classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SEED = HERE / "task_change_cases.jsonl"
OUTPUT = HERE / "task_change_train_v2.jsonl"


SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "graphql-batching",
        "anchor": "优化 GraphQL 商品详情接口的 N+1 查询。",
        "goal": "优化 GraphQL 商品详情 N+1",
        "turns": [
            {"text": "评估 DataLoader 按请求生命周期缓存的效果。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充批量加载失败时的降级和日志。", "expected": "continue", "category": "implicit_continue"},
            {"text": "设计 GraphQL schema 的权限指令系统。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计 GraphQL 权限指令"},
            {"text": "支持字段级权限，并在启动时校验指令配置。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "grpc-deadline",
        "anchor": "修复 gRPC 调用超时后上游仍在执行的问题。",
        "goal": "修复 gRPC 超时后上游继续执行",
        "turns": [
            {"text": "确认 deadline 是否正确透传到所有下游。", "expected": "continue", "category": "implicit_continue"},
            {"text": "加入 context 取消和资源释放的测试。", "expected": "continue", "category": "implicit_continue"},
            {"text": "实现 gRPC 服务间的统一鉴权拦截器。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "实现 gRPC 统一鉴权拦截器"},
            {"text": "覆盖 token 过期、权限不足和内部服务白名单。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "es-mapping",
        "anchor": "分析 Elasticsearch 字段映射导致的查询错误。",
        "goal": "修复 Elasticsearch 字段映射错误",
        "turns": [
            {"text": "检查 text 和 keyword 字段是否被混用。", "expected": "continue", "category": "implicit_continue"},
            {"text": "评估重建索引和 alias 切换步骤。", "expected": "continue", "category": "implicit_continue"},
            {"text": "为搜索集群设计冷热分层存储。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计 Elasticsearch 冷热分层"},
            {"text": "定义生命周期规则、迁移窗口和容量水位。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "mongo-transaction",
        "anchor": "排查 MongoDB 事务提交失败。",
        "goal": "排查 MongoDB 事务提交失败",
        "turns": [
            {"text": "检查写冲突、重试和事务超时配置。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充事务内外部调用的边界说明。", "expected": "continue", "category": "implicit_continue"},
            {"text": "把这份故障记录整理成对外复盘文章。", "expected": "task_changed", "category": "task_changed_cross_domain", "new_goal": "整理 MongoDB 故障复盘文章"},
            {"text": "突出时间线、影响范围和长期改进项。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "browser-memory",
        "anchor": "定位单页应用长时间使用后的内存泄漏。",
        "goal": "定位 SPA 内存泄漏",
        "turns": [
            {"text": "比较路由切换后的事件监听和组件引用。", "expected": "continue", "category": "implicit_continue"},
            {"text": "增加堆快照和泄漏回归测试。", "expected": "continue", "category": "implicit_continue"},
            {"text": "设计前端错误监控和性能采样平台。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计前端监控采样平台"},
            {"text": "控制采样率、脱敏规则和数据保留周期。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "android-anr",
        "anchor": "分析 Android 主线程 ANR。",
        "goal": "分析 Android 主线程 ANR",
        "turns": [
            {"text": "检查主线程 IO、锁等待和广播处理。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充 StrictMode 和启动性能检查。", "expected": "continue", "category": "implicit_continue"},
            {"text": "规划 Android 应用的模块化重构。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "规划 Android 模块化重构"},
            {"text": "定义模块边界、依赖规则和迁移顺序。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "websocket-reconnect",
        "anchor": "修复 WebSocket 断线重连风暴。",
        "goal": "修复 WebSocket 重连风暴",
        "turns": [
            {"text": "加入随机抖动、最大退避和服务端限速。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充网络切换和后台恢复场景。", "expected": "continue", "category": "implicit_continue"},
            {"text": "设计实时消息的离线补偿机制。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计实时消息离线补偿"},
            {"text": "说明消息游标、去重和积压上限。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "s3-upload",
        "anchor": "优化 S3 大文件上传失败率。",
        "goal": "优化 S3 大文件上传失败",
        "turns": [
            {"text": "检查分片大小、并发数和重试退避。", "expected": "continue", "category": "implicit_continue"},
            {"text": "增加校验和、断点续传和失败清理。", "expected": "continue", "category": "implicit_continue"},
            {"text": "为对象存储设计跨区域备份策略。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计对象存储跨区域备份"},
            {"text": "说明 RPO、RTO、版本控制和恢复演练。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "postgres-vacuum",
        "anchor": "分析 PostgreSQL autovacuum 跟不上写入。",
        "goal": "分析 autovacuum 落后",
        "turns": [
            {"text": "检查长事务、表膨胀和 autovacuum 参数。", "expected": "continue", "category": "implicit_continue"},
            {"text": "评估分区表和手动 vacuum 的收益。", "expected": "continue", "category": "implicit_continue"},
            {"text": "写一份数据库容量规划报告。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "编写数据库容量规划"},
            {"text": "预测未来十二个月的存储、IOPS 和连接数。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "airflow-dag",
        "anchor": "排查 Airflow DAG 每天随机失败。",
        "goal": "排查 Airflow DAG 随机失败",
        "turns": [
            {"text": "检查任务重试、资源池和依赖数据延迟。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充 SLA 监控和失败通知。", "expected": "continue", "category": "implicit_continue"},
            {"text": "为数据团队设计统一的 DAG 模板。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计 Airflow DAG 模板"},
            {"text": "定义参数、标签、告警和测试规范。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "terraform-drift",
        "anchor": "检测 Terraform 生产资源漂移。",
        "goal": "检测 Terraform 资源漂移",
        "turns": [
            {"text": "设计定时 plan 和差异分类。", "expected": "continue", "category": "implicit_continue"},
            {"text": "区分自动修复和人工确认的漂移。", "expected": "continue", "category": "implicit_continue"},
            {"text": "帮我写一份供应商年度评估模板。", "expected": "task_changed", "category": "task_changed_cross_domain", "new_goal": "编写供应商年度评估模板"},
            {"text": "按质量、交付、安全和成本评分。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "api-scopes",
        "anchor": "修复 API scope 校验遗漏。",
        "goal": "修复 API scope 校验",
        "turns": [
            {"text": "检查路由、中间件和服务层是否重复校验。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充 scope 继承和拒绝优先规则。", "expected": "continue", "category": "implicit_continue"},
            {"text": "设计面向第三方的 API key 生命周期管理。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计第三方 API key 生命周期"},
            {"text": "支持创建、轮换、过期、吊销和审计。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
]


def build() -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in SEED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for scenario in SCENARIOS:
        goal = scenario["goal"]
        for index, turn in enumerate(scenario["turns"], 1):
            case = {
                "id": f"train-v2-{scenario['name']}-{index:02d}",
                "category": turn["category"],
                "anchor": scenario["anchor"],
                "task_goal": goal,
                "latest": turn["text"],
                "expected": turn["expected"],
                "last_role": turn.get("last_role", "user"),
            }
            cases.append(case)
            if turn.get("new_goal"):
                goal = turn["new_goal"]
    return cases


def main() -> None:
    cases = build()
    OUTPUT.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
