from contextvars import ContextVar
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
user_id_var: ContextVar[str] = ContextVar("user_id", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")
client_ip_var: ContextVar[str] = ContextVar("client_ip", default="-")
method_var: ContextVar[str] = ContextVar("method", default="-")
path_var: ContextVar[str] = ContextVar("path", default="-")
status_code_var: ContextVar[str] = ContextVar("status_code", default="-")
cost_ms_var: ContextVar[str] = ContextVar("cost_ms", default="-")


def get_log_context() -> dict[str, Any]:
    return {
        "request_id": request_id_var.get(),
        "user_id": user_id_var.get(),
        "session_id": session_id_var.get(),
        "client_ip": client_ip_var.get(),
        "method": method_var.get(),
        "path": path_var.get(),
        "status_code": status_code_var.get(),
        "cost_ms": cost_ms_var.get(),
    }


def set_user_context(user_id: str | None = None, session_id: str | None = None) -> None:
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)
