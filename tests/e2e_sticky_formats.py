from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx


ENV = {}
for line in Path(
    r"C:\Users\skyro\AppData\Local\Temp\squilla_api_router\.env"
).read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        ENV[key.strip()] = value.strip()

AUTH = {
    "Authorization": "Bearer " + ENV["ROUTER_API_KEY"],
    "Content-Type": "application/json",
}


def run(
    url: str, body: dict, session: str, extra_headers: dict[str, str] | None = None
) -> dict:
    headers = dict(AUTH)
    headers["X-Squilla-Session-Id"] = session
    if extra_headers:
        headers.update(extra_headers)
    with httpx.Client(trust_env=False, timeout=90) as client:
        with client.stream("POST", url, headers=headers, json=body) as response:
            for _ in response.iter_lines():
                pass
            return {
                key: response.headers.get(key)
                for key in (
                    "x-router-tier",
                    "x-router-model",
                    "x-squilla-intent",
                    "x-squilla-intent-reason",
                    "x-squilla-sticky",
                    "x-squilla-task-id",
                    "x-squilla-classifier-tier",
                    "x-squilla-cost-aware",
                    "x-squilla-cost-policy",
                )
            }


def main() -> None:
    formats = [
        (
            "responses",
            "http://127.0.0.1:8021/v1/responses",
            lambda prompt: {
                "model": "router",
                "stream": True,
                "max_output_tokens": 64,
                "input": prompt,
            },
        ),
        (
            "anthropic",
            "http://127.0.0.1:8021/v1/messages",
            lambda prompt: {
                "model": "router",
                "stream": True,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": prompt}],
            },
        ),
    ]
    for label, url, make_body in formats:
        session = f"e2e-{label}-{uuid.uuid4().hex[:8]}"
        first = run(url, make_body("把 hello world 翻译成中文。"), session)
        second = run(
            url,
            make_body("请继续，把这个翻译改得更正式一些。"),
            session,
        )
        print(
            json.dumps(
                {"format": label, "first": first, "second": second},
                ensure_ascii=False,
            ),
            flush=True,
        )

    session = f"e2e-redo-{uuid.uuid4().hex[:8]}"
    chat_url = "http://127.0.0.1:8021/v1/chat/completions"
    chat_body = lambda prompt: {  # noqa: E731
        "model": "router",
        "stream": True,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    }
    first = run(chat_url, chat_body("把 hello world 翻译成中文。"), session)
    upgraded = run(chat_url, chat_body("重新生成一次。"), session)
    with httpx.Client(trust_env=False, timeout=30) as client:
        state = client.get(
            "http://127.0.0.1:8021/admin/routing-state",
            headers=AUTH,
            params={"session_id": session},
        ).json()
        usage = client.get(
            "http://127.0.0.1:8021/admin/routing-usage",
            headers=AUTH,
            params={"session_id": session},
        ).json()
    print(
        json.dumps(
            {
                "format": "chat_redo",
                "first": first,
                "upgraded": upgraded,
                "state": state,
                "usage": usage,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    quality_session = f"e2e-quality-{uuid.uuid4().hex[:8]}"
    quality_first = run(
        chat_url,
        chat_body("判断 hello 是否回文。"),
        quality_session,
    )
    quality_retry = run(
        chat_url,
        chat_body("请修复上一次结果中的错误。"),
        quality_session,
        {"X-Squilla-Quality-Feedback": "validation_failed"},
    )
    print(
        json.dumps(
            {
                "format": "quality_feedback",
                "first": quality_first,
                "retry": quality_retry,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    cost_session = f"e2e-cost-{uuid.uuid4().hex[:8]}"
    cost_body = lambda prompt: {  # noqa: E731
        "model": "router",
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    cost_first = run(
        chat_url,
        cost_body("写一个 Python 脚本读取 JSON 文件并打印指定字段。"),
        cost_session,
    )
    cost_continue = run(
        chat_url,
        cost_body("请继续，为这个脚本增加错误处理。"),
        cost_session,
    )
    cost_state = httpx.get(
        "http://127.0.0.1:8021/admin/routing-state",
        headers=AUTH,
        params={"session_id": cost_session},
        timeout=30,
    ).json()
    classifier_session = f"e2e-classifier-{uuid.uuid4().hex[:8]}"
    classifier_first = run(
        chat_url,
        cost_body("写一个 Python 脚本读取 JSON 文件并打印指定字段。"),
        classifier_session,
        {"X-Squilla-Cost-Policy": "classifier"},
    )
    print(
        json.dumps(
            {
                "format": "cost_aware",
                "first": cost_first,
                "continue": cost_continue,
                "classifier_only": classifier_first,
                "state": cost_state,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
