from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from starlette.requests import Request

import app as router_app
from squilla_api_router.routing_state import resolve_request_identity


def _request(headers: dict[str, str] | None = None) -> Request:
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": raw_headers,
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8021),
        }
    )


def _bearer(value: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=value)


def test_tenant_key_sets_authoritative_session_namespace(monkeypatch) -> None:
    monkeypatch.setattr(router_app, "ROUTER_API_KEY", "global-key")
    monkeypatch.setattr(
        router_app,
        "ROUTER_TENANT_API_KEYS",
        {"tenant-a": "tenant-a-key"},
    )
    request = _request()

    principal = router_app._authenticate(request, _bearer("tenant-a-key"))

    assert principal.tenant_id == "tenant-a"
    assert principal.is_admin is False
    identity = resolve_request_identity(
        router_app._headers_with_authenticated_tenant(request),
        {
            "metadata": {"tenant_id": "tenant-b"},
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert identity.session_id.startswith("tenant-a:")
    assert "tenant-b" not in identity.session_id


def test_tenant_header_cannot_override_bound_key(monkeypatch) -> None:
    monkeypatch.setattr(router_app, "ROUTER_API_KEY", "global-key")
    monkeypatch.setattr(
        router_app,
        "ROUTER_TENANT_API_KEYS",
        {"tenant-a": "tenant-a-key"},
    )
    request = _request({"x-squilla-tenant-id": "tenant-b"})

    with pytest.raises(HTTPException) as exc_info:
        router_app._authenticate(request, _bearer("tenant-a-key"))

    assert exc_info.value.status_code == 403


def test_global_key_is_admin_and_can_select_tenant(monkeypatch) -> None:
    monkeypatch.setattr(router_app, "ROUTER_API_KEY", "global-key")
    monkeypatch.setattr(router_app, "ROUTER_TENANT_API_KEYS", {})
    request = _request({"x-squilla-tenant-id": "tenant-z"})

    principal = router_app._authenticate(request, _bearer("global-key"))

    assert principal.is_admin is True
    assert principal.tenant_id == "tenant-z"
    assert router_app._verify_admin_auth(principal) == principal


def test_admin_endpoint_rejects_tenant_key(monkeypatch) -> None:
    monkeypatch.setattr(router_app, "ROUTER_API_KEY", "global-key")
    monkeypatch.setattr(
        router_app,
        "ROUTER_TENANT_API_KEYS",
        {"tenant-a": "tenant-a-key"},
    )
    with TestClient(router_app.app) as client:
        denied = client.post(
            "/admin/purge-state",
            headers={"Authorization": "Bearer tenant-a-key"},
        )
        allowed = client.post(
            "/admin/purge-state",
            headers={"Authorization": "Bearer global-key"},
        )

    assert denied.status_code == 403
    assert allowed.status_code == 200


def test_missing_and_invalid_keys_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr(router_app, "ROUTER_API_KEY", "global-key")
    monkeypatch.setattr(
        router_app,
        "ROUTER_TENANT_API_KEYS",
        {"tenant-a": "tenant-a-key"},
    )

    with pytest.raises(HTTPException) as missing:
        router_app._authenticate(_request(), None)
    with pytest.raises(HTTPException) as invalid:
        router_app._authenticate(_request(), _bearer("wrong-key"))

    assert missing.value.status_code == 401
    assert invalid.value.status_code == 401


def test_tenant_mapping_rejects_global_key_and_unsafe_tenant_id(
    monkeypatch,
) -> None:
    monkeypatch.setattr(router_app, "ROUTER_API_KEY", "global-key")
    tenants = router_app._load_tenant_api_keys(
        '{"tenant-a":"global-key","bad:tenant":"tenant-key","tenant-b":"key-b"}'
    )

    assert tenants == {"tenant-b": "key-b"}
