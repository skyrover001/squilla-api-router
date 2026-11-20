"""Build the expanded multi-turn task-change holdout from fixed scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SEED = HERE / "task_change_holdout.jsonl"
OUTPUT = HERE / "task_change_holdout_v2.jsonl"


SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "oauth-refresh",
        "anchor": "修复 OAuth refresh token 偶发失效的问题。",
        "goal": "修复 OAuth refresh token 偶发失效",
        "turns": [
            {"text": "先检查并发刷新时旧 token 是否被提前吊销。", "expected": "continue", "category": "implicit_continue"},
            {"text": "再补充刷新失败时的重试和告警策略。", "expected": "continue", "category": "implicit_continue"},
            {"text": "设计登录设备管理和远程会话踢出功能。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计登录设备管理与会话踢出"},
            {"text": "增加最近登录地点、设备类型和最后活跃时间。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "postgres-audit",
        "anchor": "优化 PostgreSQL 审计表写入性能。",
        "goal": "优化 PostgreSQL 审计表写入性能",
        "turns": [
            {"text": "检查索引数量是否导致写放大。", "expected": "continue", "category": "implicit_continue"},
            {"text": "评估批量写入和异步归档的收益。", "expected": "continue", "category": "implicit_continue"},
            {"text": "为审计日志设计按月分区的迁移方案。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计审计日志按月分区迁移"},
            {"text": "补充历史数据回填、锁表时间和回滚步骤。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "react-form",
        "anchor": "修复 React 结算表单输入卡顿。",
        "goal": "修复 React 结算表单输入卡顿",
        "turns": [
            {"text": "检查是否每次输入都触发整棵表单重渲染。", "expected": "continue", "category": "implicit_continue"},
            {"text": "对比 useMemo、受控字段拆分和外部状态方案。", "expected": "continue", "category": "implicit_continue"},
            {"text": "为 React 项目设计统一的表单组件库。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计 React 统一表单组件库"},
            {"text": "先定义校验、布局、可访问性和错误展示接口。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "nginx-502",
        "anchor": "排查 Nginx 到上游服务的偶发 502。",
        "goal": "排查 Nginx 偶发 502",
        "turns": [
            {"text": "核对 upstream keepalive 和后端超时参数。", "expected": "continue", "category": "implicit_continue"},
            {"text": "检查重试是否把瞬时故障放大成雪崩。", "expected": "continue", "category": "implicit_continue"},
            {"text": "帮我规划下周去成都的三天旅行。", "expected": "task_changed", "category": "task_changed_cross_domain", "new_goal": "规划成都三天旅行"},
            {"text": "酒店放在地铁附近，并控制总预算。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "k8s-rollout",
        "anchor": "设计 Kubernetes 灰度发布流程。",
        "goal": "设计 Kubernetes 灰度发布流程",
        "turns": [
            {"text": "加入 5%、20%、50%、100% 的流量阶段。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充指标异常时的自动暂停和回滚。", "expected": "continue", "category": "implicit_continue"},
            {"text": "部署一套跨机房 Kubernetes 灾备集群。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "部署跨机房 Kubernetes 灾备集群"},
            {"text": "说明 etcd、存储和 DNS 的故障切换顺序。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "redis-memory",
        "anchor": "分析 Redis 内存持续增长。",
        "goal": "分析 Redis 内存持续增长",
        "turns": [
            {"text": "检查大 key、碎片率和过期键比例。", "expected": "continue", "category": "implicit_continue"},
            {"text": "比较主动淘汰和按业务拆分实例。", "expected": "continue", "category": "implicit_continue"},
            {"text": "把这份故障分析写成给管理层的邮件。", "expected": "task_changed", "category": "task_changed_cross_domain", "new_goal": "撰写 Redis 故障管理层邮件"},
            {"text": "语气客观，突出影响、根因和后续动作。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "kafka-lag",
        "anchor": "定位 Kafka consumer group 的消息堆积。",
        "goal": "定位 Kafka consumer group 消息堆积",
        "turns": [
            {"text": "先区分生产突增、消费变慢和 rebalance。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充每个分区 lag 和消费耗时的监控。", "expected": "continue", "category": "implicit_continue"},
            {"text": "实现消息消费失败的死信队列处理服务。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "实现消费失败死信队列"},
            {"text": "支持最大重试次数、退避和人工重新投递。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "terraform-iam",
        "anchor": "Review Terraform IAM changes for privilege escalation.",
        "goal": "Review Terraform IAM privilege escalation",
        "turns": [
            {"text": "Focus on wildcard actions and trust policy changes.", "expected": "continue", "category": "implicit_continue", "last_role": "user"},
            {"text": "Include cross-account role assumption risks.", "expected": "continue", "category": "implicit_continue"},
            {"text": "Write an IAM permission matrix for the new support team.", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "Write IAM permission matrix for support team"},
            {"text": "Separate read-only access from approval-required operations.", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "csv-import",
        "anchor": "排查 CSV 导入任务的重复记录。",
        "goal": "排查 CSV 导入重复记录",
        "turns": [
            {"text": "检查批次重试时幂等键是否稳定。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充重复数据和脏数据的核对报告。", "expected": "continue", "category": "implicit_continue"},
            {"text": "写一个把 CSV 批量转换成 JSON 的 CLI。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "实现 CSV 批量转 JSON CLI"},
            {"text": "支持流式读取、字段筛选和错误行输出。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "search-relevance",
        "anchor": "优化商品搜索的相关性。",
        "goal": "优化商品搜索相关性",
        "turns": [
            {"text": "检查标题、类目和销量特征的权重。", "expected": "continue", "category": "implicit_continue"},
            {"text": "增加零结果查询和长尾查询的评估集。", "expected": "continue", "category": "implicit_continue"},
            {"text": "实现商品搜索的多语言同义词服务。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "实现搜索多语言同义词服务"},
            {"text": "支持运营审核、版本发布和回滚。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "mobile-lcp",
        "anchor": "优化移动端首页 LCP。",
        "goal": "优化移动端首页 LCP",
        "turns": [
            {"text": "分析首屏图片、字体和主线程阻塞。", "expected": "continue", "category": "implicit_continue"},
            {"text": "比较预加载、CDN 和图片格式转换的收益。", "expected": "continue", "category": "implicit_continue"},
            {"text": "生成一份家庭年度预算表模板。", "expected": "task_changed", "category": "task_changed_cross_domain", "new_goal": "生成家庭年度预算表"},
            {"text": "按住房、教育、保险、旅行和储蓄分类。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "webhook-signature",
        "anchor": "检查 webhook 签名校验的安全问题。",
        "goal": "检查 webhook 签名校验安全",
        "turns": [
            {"text": "确认时间戳、重放保护和密钥轮换。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充不同语言 SDK 的签名示例。", "expected": "continue", "category": "implicit_continue"},
            {"text": "实现 webhook 投递重试和死信处理。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "实现 webhook 投递重试与死信处理"},
            {"text": "支持指数退避、目标禁用和手工重放。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "ci-build",
        "anchor": "排查 CI 构建在测试阶段超时。",
        "goal": "排查 CI 构建测试超时",
        "turns": [
            {"text": "比较测试分片、缓存和并发任务的耗时。", "expected": "continue", "category": "implicit_continue"},
            {"text": "找出最慢的十个测试并列出优化建议。", "expected": "continue", "category": "implicit_continue"},
            {"text": "为工程团队写一份季度技术债务治理计划。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "制定季度技术债务治理计划"},
            {"text": "按风险、收益、成本和负责人排出优先级。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "docker-size",
        "anchor": "缩小 Docker 生产镜像体积。",
        "goal": "缩小 Docker 镜像体积",
        "turns": [
            {"text": "检查基础镜像、构建缓存和多阶段构建。", "expected": "continue", "category": "implicit_continue"},
            {"text": "比较 distroless 和 Alpine 的兼容性。", "expected": "continue", "category": "implicit_continue"},
            {"text": "写一份 Docker 镜像的供应链安全规范。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "编写 Docker 镜像供应链安全规范"},
            {"text": "覆盖 SBOM、签名、漏洞扫描和准入控制。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "api-pagination",
        "anchor": "修复列表 API 分页重复数据。",
        "goal": "修复列表 API 分页重复数据",
        "turns": [
            {"text": "检查排序字段是否唯一且稳定。", "expected": "continue", "category": "implicit_continue"},
            {"text": "比较 offset 分页和 cursor 分页。", "expected": "continue", "category": "implicit_continue"},
            {"text": "解释一下什么是向量数据库以及它的适用场景。", "expected": "task_changed", "category": "task_changed_cross_domain", "new_goal": "解释向量数据库及适用场景"},
            {"text": "重点说明索引类型、召回率和成本。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "data-retention",
        "anchor": "设计日志数据的保留和删除策略。",
        "goal": "设计日志数据保留删除策略",
        "turns": [
            {"text": "按数据分类、合规要求和查询频率分层。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充删除失败、法务保留和审计要求。", "expected": "continue", "category": "implicit_continue"},
            {"text": "为客服团队编写退款政策说明。", "expected": "task_changed", "category": "task_changed_cross_domain", "new_goal": "编写客服退款政策说明"},
            {"text": "明确时限、例外、审批和用户沟通模板。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "file-upload",
        "anchor": "定位大文件上传失败的问题。",
        "goal": "定位大文件上传失败",
        "turns": [
            {"text": "检查代理 body 限制、分片大小和超时。", "expected": "continue", "category": "implicit_continue"},
            {"text": "补充断点续传和分片合并校验。", "expected": "continue", "category": "implicit_continue"},
            {"text": "设计文件上传完成后的病毒扫描流水线。", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "设计上传文件病毒扫描流水线"},
            {"text": "定义隔离区、扫描超时和人工释放流程。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "translation-doc",
        "anchor": "把发布说明翻译成英文。",
        "goal": "把发布说明翻译成英文",
        "turns": [
            {"text": "统一版本、回滚和兼容性相关术语。", "expected": "continue", "category": "implicit_continue"},
            {"text": "保持正式、简洁的技术文案风格。", "expected": "continue", "category": "implicit_continue"},
            {"text": "重做，面向普通用户再写一版更通俗的说明。", "expected": "redo", "category": "explicit_redo"},
            {"text": "减少术语，增加升级前注意事项。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "interview-prep",
        "anchor": "整理后端工程师面试题。",
        "goal": "整理后端工程师面试题",
        "turns": [
            {"text": "按数据库、缓存、分布式和系统设计分类。", "expected": "continue", "category": "implicit_continue"},
            {"text": "为每个问题补充优秀回答和常见误区。", "expected": "continue", "category": "implicit_continue"},
            {"text": "现在做另一个任务，准备一份薪资谈判话术。", "expected": "task_changed", "category": "explicit_new_task", "new_goal": "准备薪资谈判话术"},
            {"text": "覆盖报价、竞争 offer 和拒绝低薪的方案。", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "meal-plan",
        "anchor": "制定一周低盐饮食计划。",
        "goal": "制定一周低盐饮食计划",
        "turns": [
            {"text": "每天热量控制在 1800 千卡左右。", "expected": "continue", "category": "implicit_continue"},
            {"text": "列出采购清单和可提前准备的食材。", "expected": "continue", "category": "implicit_continue"},
            {"text": "解释一下什么是光合作用。", "expected": "task_changed", "category": "task_changed_cross_domain", "new_goal": "解释光合作用"},
            {"text": "再详细一点，说明光反应和暗反应。", "expected": "continue", "category": "terse_continue"},
        ],
    },
    {
        "name": "contract-review",
        "anchor": "检查租房合同中的风险条款。",
        "goal": "检查租房合同风险条款",
        "turns": [
            {"text": "重点看押金、提前退租和维修责任。", "expected": "continue", "category": "implicit_continue"},
            {"text": "列出需要和房东协商修改的条款。", "expected": "continue", "category": "implicit_continue"},
            {"text": "另外做一个任务：规划一次周末露营。", "expected": "task_changed", "category": "explicit_new_task", "new_goal": "规划周末露营"},
            {"text": "预算控制在两千元以内，并考虑雨天备案。", "expected": "continue", "category":"implicit_continue"},
        ],
    },
    {
        "name": "rust-parser",
        "anchor": "Review the Rust parser for panic paths.",
        "goal": "Review Rust parser panic paths",
        "turns": [
            {"text": "Check malformed UTF-8 and truncated input cases.", "expected": "continue", "category": "implicit_continue"},
            {"text": "Suggest error types that preserve source positions.", "expected": "continue", "category": "implicit_continue"},
            {"text": "Build a fuzzing harness for the same parser.", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "Build parser fuzzing harness"},
            {"text": "Include corpus seeding, minimization, and CI time budgets.", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "go-http",
        "anchor": "Fix goroutine leaks in the Go HTTP service.",
        "goal": "Fix Go HTTP goroutine leaks",
        "turns": [
            {"text": "Inspect request cancellation and response body closure.", "expected": "continue", "category": "implicit_continue"},
            {"text": "Add a goroutine count regression test.", "expected": "continue", "category": "implicit_continue"},
            {"text": "Design a graceful shutdown sequence for the service.", "expected": "task_changed", "category": "task_changed_shared_domain", "new_goal": "Design Go service graceful shutdown"},
            {"text": "Cover listener close, in-flight requests, and background workers.", "expected": "continue", "category": "implicit_continue"},
        ],
    },
    {
        "name": "tool-contract",
        "anchor": "分析慢查询日志并给出索引建议。",
        "goal": "分析慢查询日志并给出索引建议",
        "turns": [
            {"text": "先按查询模板聚合耗时和扫描行数。", "expected": "continue", "category": "implicit_continue"},
            {"text": "切换到新的只读数据库工具集继续分析。", "expected": "task_changed", "category": "session_contract_change", "tools_changed": True, "new_goal": "在新只读工具集下重新分析慢查询"},
            {"text": "确认新工具看不到锁等待，需要调整采集方式。", "expected": "continue", "category": "implicit_continue"},
            {"text": "{\"exit_code\":0,\"rows\":18}", "expected": "continue", "category": "tool_continuation", "last_role": "tool"},
        ],
    },
]


def build() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seed_lines = SEED.read_text(encoding="utf-8").splitlines()
    cases.extend(json.loads(line) for line in seed_lines if line.strip())
    for scenario in SCENARIOS:
        goal = scenario["goal"]
        for index, turn in enumerate(scenario["turns"], 1):
            case = {
                "id": f"v2-{scenario['name']}-{index:02d}",
                "category": turn["category"],
                "anchor": scenario["anchor"],
                "task_goal": goal,
                "latest": turn["text"],
                "expected": turn["expected"],
                "last_role": turn.get("last_role", "user"),
            }
            for key in ("tools_changed", "system_changed", "explicit_intent"):
                if key in turn:
                    case[key] = turn[key]
            cases.append(case)
            if turn.get("new_goal"):
                goal = turn["new_goal"]
    return cases


def main() -> None:
    cases = build()
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case ids")
    OUTPUT.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
