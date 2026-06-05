"""Thin async wrapper for sending ntfy push notifications.

Reads ntfy connection details from the Odysseus integrations system
(the same ntfy entry the reminder channel uses), with a fallback to
the NTFY_BASE_URL / NTFY_TOPIC_PREFIX environment variables.

Usage:
    from src.notifications.ntfy import send

    ok = await send(
        topic="assistant/health",
        message="Time to take Lisinopril 10mg",
        title="Medication Reminder",
        priority="high",
        tags=["pill"],
    )
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = os.getenv("NTFY_BASE_URL", "http://ntfy:80")
_TOPIC_PREFIX = os.getenv("NTFY_TOPIC_PREFIX", "assistant")


def _resolve_ntfy_config() -> tuple[str, str]:
    """Return (base_url, api_key) from integrations or env fallback."""
    try:
        from src.integrations import load_integrations
        intg = next(
            (i for i in load_integrations()
             if i.get("preset") == "ntfy" and i.get("enabled", True) and i.get("base_url")),
            None,
        )
        if intg:
            return intg["base_url"].rstrip("/"), intg.get("api_key", "")
    except Exception as e:
        logger.debug("ntfy: integrations lookup failed, using env fallback: %s", e)
    return _DEFAULT_BASE_URL.rstrip("/"), ""


async def send(
    topic: str,
    message: str,
    title: Optional[str] = None,
    priority: str = "default",
    tags: Optional[list] = None,
    extra_headers: Optional[dict] = None,
) -> bool:
    """Send an ntfy push notification. Returns True on success.

    extra_headers can include ntfy-specific headers such as ``Actions``,
    ``Click``, ``Attach``, etc.
    """
    base_url, api_key = _resolve_ntfy_config()

    # Support bare topic names and full topic paths
    if not topic.startswith(("http://", "https://")):
        url = f"{base_url}/{topic}"
    else:
        url = topic

    headers: dict = {}
    if title:
        headers["Title"] = title
    if priority and priority != "default":
        headers["Priority"] = priority
    if tags:
        headers["Tags"] = ",".join(tags)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, content=message.encode(), headers=headers)
            if resp.is_success:
                logger.debug("ntfy: sent to %s (HTTP %s)", url, resp.status_code)
                return True
            logger.warning("ntfy: POST %s returned HTTP %s", url, resp.status_code)
            return False
    except Exception as e:
        logger.warning("ntfy: send failed: %s", e)
        return False


def topic(name: str) -> str:
    """Build a full topic path from a short name, e.g. 'health' → 'assistant/health'."""
    prefix = _TOPIC_PREFIX.strip("/")
    name = name.strip("/")
    return f"{prefix}/{name}" if prefix else name
