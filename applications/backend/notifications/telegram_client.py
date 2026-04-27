from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from .config import NotificationConfig
from .models import DeliveryResult


logger = logging.getLogger(__name__)


class TelegramClient:
    """Minimal Telegram Bot API sender."""

    def __init__(self, config: NotificationConfig):
        self._cfg = config

    def send_message(self, *, chat_id: str, text: str) -> DeliveryResult:
        if not (self._cfg.telegram_bot_token and chat_id):
            return DeliveryResult(channel="telegram", attempted=False, status="skipped_missing_config", detail="")

        payload = json.dumps(
            {
                "chat_id": str(chat_id),
                "text": str(text),
                "disable_web_page_preview": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._cfg.telegram_api_base}/bot{self._cfg.telegram_bot_token}/sendMessage",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.http_timeout_s) as resp:
                if int(getattr(resp, "status", 200)) >= 300:
                    raise urllib.error.HTTPError(req.full_url, resp.status, "telegram send failed", hdrs=resp.headers, fp=None)
            return DeliveryResult(channel="telegram", attempted=True, status="sent", detail="")
        except urllib.error.HTTPError as exc:
            detail = str(exc)
            try:
                raw = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                raw = ""
            if raw:
                detail = f"{detail} body={raw}"
            logger.warning("telegram send failed: %s", detail)
            return DeliveryResult(channel="telegram", attempted=True, status="failed", detail=detail)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("telegram send failed: %s", exc)
            return DeliveryResult(channel="telegram", attempted=True, status="failed", detail=str(exc))
