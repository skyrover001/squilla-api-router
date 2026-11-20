from __future__ import annotations

import os

import app as router_app


def _clear_provider_env(monkeypatch) -> None:
    for key in list(os.environ):
        if key.upper().endswith("_BASE_URL"):
            monkeypatch.delenv(key, raising=False)


def test_explicit_provider_namespace_is_canonical(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv(
        "PROVIDER_PRIMARY_BASE_URL",
        "https://chat.example/v1/",
    )
    monkeypatch.setenv("PROVIDER_PRIMARY_API_KEY", "key")
    monkeypatch.setenv("PROVIDER_PRIMARY_FORMAT", "openai_chat")

    providers = router_app._discover_providers()

    assert providers["primary"] == {
        "base_url": "https://chat.example/v1",
        "api_key": "key",
        "format": "openai_chat",
    }


def test_backend_prefix_has_no_hidden_alias(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("BACKEND_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("BACKEND_API_KEY", "legacy-key")
    monkeypatch.setenv("BACKEND_FORMAT", "openai_chat")

    providers = router_app._discover_providers()

    assert set(providers) == {"backend"}
    assert "primary" not in providers


def test_explicit_provider_wins_over_generic_prefix(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PRIMARY_BASE_URL", "https://generic.example/v1")
    monkeypatch.setenv("PRIMARY_API_KEY", "generic")
    monkeypatch.setenv(
        "PROVIDER_PRIMARY_BASE_URL",
        "https://explicit.example/v1",
    )
    monkeypatch.setenv("PROVIDER_PRIMARY_API_KEY", "explicit")

    providers = router_app._discover_providers()

    assert providers["primary"]["base_url"] == "https://explicit.example/v1"
    assert providers["primary"]["api_key"] == "explicit"


def test_format_is_inferred_from_anthropic_path(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv(
        "PROVIDER_DEEPSEEK_BASE_URL",
        "https://api.example/anthropic",
    )

    providers = router_app._discover_providers()

    assert providers["deepseek"]["format"] == "anthropic"
