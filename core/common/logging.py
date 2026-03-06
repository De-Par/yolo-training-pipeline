from __future__ import annotations


def format_log(level: str, message: str) -> str:
    return f"[{level}] {message}"


def format_info(message: str) -> str:
    return format_log("INFO", message)


def format_warning(message: str) -> str:
    return format_log("WARN", message)


def format_error(message: str) -> str:
    return format_log("ERROR", message)


def format_hint(message: str) -> str:
    return format_log("HINT", message)


def format_detail(message: str) -> str:
    return format_log("DETAIL", message)
