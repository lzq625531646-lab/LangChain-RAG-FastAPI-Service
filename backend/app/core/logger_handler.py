import logging
import logging.config
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.core.request_context import get_log_context

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
_log_dir = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR = _log_dir if _log_dir.is_absolute() else BASE_DIR / _log_dir
LOG_DIR = LOG_DIR.resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "14"))
LOG_WHEN = os.getenv("LOG_ROTATE_WHEN", "midnight")
LOG_INTERVAL = int(os.getenv("LOG_ROTATE_INTERVAL", "1"))

LOG_FORMAT = (
    "%(asctime)s - %(levelname)s - %(name)s - "
    "request_id=%(request_id)s user_id=%(user_id)s session_id=%(session_id)s "
    "client_ip=%(client_ip)s method=%(method)s path=%(path)s "
    "status=%(status_code)s cost_ms=%(cost_ms)s "
    "%(filename)s:%(lineno)d - %(message)s"
)


class RequestContextFilter(logging.Filter):
    """为所有日志记录补齐请求上下文字段，避免 formatter 因缺字段报错。"""

    def filter(self, record: logging.LogRecord) -> bool:
        context = get_log_context()
        for key, value in context.items():
            setattr(record, key, value)
        return True


class AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "app.access"


class ExcludeAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != "app.access"


def _file_handler(filename: str, level: str = "DEBUG") -> dict[str, Any]:
    return {
        "class": "logging.handlers.TimedRotatingFileHandler",
        "level": level,
        "formatter": "default",
        "filters": ["request_context"],
        "filename": str(LOG_DIR / filename),
        "when": LOG_WHEN,
        "interval": LOG_INTERVAL,
        "backupCount": LOG_BACKUP_COUNT,
        "encoding": "utf-8",
        "utc": False,
    }


def setup_logging() -> None:
    """初始化本地文件日志。

    默认生成：
    - app.log：业务与框架普通日志
    - error.log：ERROR及以上日志
    - access.log：HTTP访问日志
    """

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "request_context": {"()": RequestContextFilter},
                "access_only": {"()": AccessLogFilter},
                "exclude_access": {"()": ExcludeAccessLogFilter},
            },
            "formatters": {
                "default": {
                    "format": LOG_FORMAT,
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": LOG_LEVEL,
                    "formatter": "default",
                    "filters": ["request_context"],
                },
                "app_file": {
                    **_file_handler("app.log", "DEBUG"),
                    "filters": ["request_context", "exclude_access"],
                },
                "error_file": _file_handler("error.log", "ERROR"),
                "access_file": {
                    **_file_handler("access.log", "INFO"),
                    "filters": ["request_context", "access_only"],
                },
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["console", "app_file", "error_file", "access_file"],
            },
            "loggers": {
                "uvicorn": {"level": LOG_LEVEL, "propagate": True},
                "uvicorn.access": {"level": LOG_LEVEL, "propagate": True},
                "sqlalchemy.engine": {
                    "level": os.getenv("SQL_LOG_LEVEL", "WARNING").upper(),
                    "propagate": True,
                },
                "asyncio": {"level": os.getenv("ASYNCIO_LOG_LEVEL", "WARNING").upper(), "propagate": True},
                "httpx": {"level": os.getenv("HTTPX_LOG_LEVEL", "WARNING").upper(), "propagate": True},
            },
        }
    )


def get_logger(name: str = "agent") -> logging.Logger:
    return logging.getLogger(name)


setup_logging()
logger = get_logger("agent")


if __name__ == "__main__":
    logger.info("日志系统初始化完成，日志目录: %s", LOG_DIR)
    logger.debug("这是一条debug日志")
    logger.error("这是一条error日志")
