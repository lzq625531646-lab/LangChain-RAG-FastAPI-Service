import os
import time
import uuid
import json
from typing import Any

from app.core.logger_handler import get_logger
from app.core.request_context import (
    client_ip_var,
    cost_ms_var,
    method_var,
    path_var,
    request_id_var,
    session_id_var,
    status_code_var,
    user_id_var,
)

access_logger = get_logger("app.access")

MAX_BODY_LOG_LENGTH = int(os.getenv("MAX_BODY_LOG_LENGTH", "4000"))
LOG_REQUEST_BODY = os.getenv("LOG_REQUEST_BODY", "true").lower() == "true"


def _header(headers: list[tuple[bytes, bytes]], name: str) -> str:
    name_bytes = name.lower().encode()
    for key, value in headers:
        if key.lower() == name_bytes:
            return value.decode("latin-1")
    return ""


def _client_ip(scope: dict[str, Any]) -> str:
    headers = scope.get("headers", [])
    forwarded_for = _header(headers, "x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    client = scope.get("client")
    if client:
        return client[0]
    return "unknown"


def _query_string(scope: dict[str, Any]) -> str:
    raw_query = scope.get("query_string", b"")
    return raw_query.decode("utf-8", errors="replace")


def _body_preview(chunks: list[bytes]) -> str:
    if not chunks:
        return ""
    body = b"".join(chunks)
    text = body.decode("utf-8", errors="replace")
    if len(text) > MAX_BODY_LOG_LENGTH:
        return f"{text[:MAX_BODY_LOG_LENGTH]}...<truncated {len(text) - MAX_BODY_LOG_LENGTH} chars>"
    return text


def _session_id_from_body(chunks: list[bytes]) -> str | None:
    if not chunks:
        return None
    try:
        data = json.loads(b"".join(chunks).decode("utf-8", errors="replace"))
    except Exception:
        return None
    if isinstance(data, dict):
        session_id = data.get("session_id")
        if session_id:
            return str(session_id)
    return None


class RequestLoggingMiddleware:
    """纯 ASGI 请求日志中间件，兼容 StreamingResponse/SSE。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        request_id = _header(headers, "x-request-id") or str(uuid.uuid4())
        query = _query_string(scope)
        session_id = "-"
        for part in query.split("&"):
            if part.startswith("session_id="):
                session_id = part.split("=", 1)[1] or "-"
                break

        content_type = _header(headers, "content-type") or "-"
        content_length = _header(headers, "content-length") or "-"
        method = scope.get("method", "-")
        path = scope.get("path", "-")

        tokens = [
            request_id_var.set(request_id),
            user_id_var.set("-"),
            session_id_var.set(session_id),
            client_ip_var.set(_client_ip(scope)),
            method_var.set(method),
            path_var.set(path),
            status_code_var.set("-"),
            cost_ms_var.set("-"),
        ]

        body_chunks: list[bytes] = []
        should_log_body = (
            LOG_REQUEST_BODY
            and method in {"POST", "PUT", "PATCH"}
            and "multipart/form-data" not in content_type
        )
        status_code = 500
        response_content_length = "-"
        start = time.perf_counter()

        access_logger.info(
            "request started query=%s content_type=%s content_length=%s",
            query,
            content_type,
            content_length,
        )

        async def receive_wrapper():
            message = await receive()
            if should_log_body and message["type"] == "http.request":
                chunk = message.get("body", b"")
                if chunk:
                    body_chunks.append(chunk)
                if not message.get("more_body", False):
                    body_session_id = _session_id_from_body(body_chunks)
                    if body_session_id:
                        session_id_var.set(body_session_id)
            return message

        async def send_wrapper(message):
            nonlocal status_code, response_content_length
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
                status_code_var.set(str(status_code))
                response_headers = message.setdefault("headers", [])
                response_headers.append((b"x-request-id", request_id.encode()))
                response_headers.append((b"x-process-time", str(round((time.perf_counter() - start), 4)).encode()))
                response_content_length = _header(response_headers, "content-length") or "-"
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception:
            status_code_var.set("500")
            access_logger.exception("request failed body=%s", _body_preview(body_chunks))
            raise
        finally:
            cost_ms = round((time.perf_counter() - start) * 1000, 2)
            status_code_var.set(str(status_code))
            cost_ms_var.set(str(cost_ms))
            access_logger.info(
                "request completed status=%s cost_ms=%s response_content_length=%s body=%s",
                status_code,
                cost_ms,
                response_content_length,
                _body_preview(body_chunks),
            )

            for var, token in zip(
                [
                    request_id_var,
                    user_id_var,
                    session_id_var,
                    client_ip_var,
                    method_var,
                    path_var,
                    status_code_var,
                    cost_ms_var,
                ],
                tokens,
            ):
                var.reset(token)
