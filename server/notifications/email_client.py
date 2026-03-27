from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from email.message import EmailMessage

from .config import NotificationConfig
from .models import DeliveryResult


logger = logging.getLogger(__name__)


class EmailClient:
    """Resend email sender."""

    def __init__(self, config: NotificationConfig):
        self._cfg = config

    def send(self, msg: EmailMessage) -> DeliveryResult:
        to_addr = str(msg.get("To") or "").strip()
        if not (self._cfg.resend_api_key and self._cfg.email_from and to_addr):
            return DeliveryResult(channel="email", attempted=False, status="skipped_missing_config", detail="")

        payload = json.dumps(
            {
                "from": self._cfg.email_from,
                "to": [to_addr],
                "subject": str(msg.get("Subject") or "").strip(),
                "text": msg.get_content(),
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._cfg.resend_api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.http_timeout_s) as resp:
                if int(getattr(resp, "status", 200)) >= 300:
                    raise urllib.error.HTTPError(req.full_url, resp.status, "resend send failed", hdrs=resp.headers, fp=None)
            return DeliveryResult(channel="email", attempted=True, status="sent", detail="")
        except urllib.error.HTTPError as exc:
            detail = str(exc)
            try:
                raw = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                raw = ""
            if raw:
                detail = f"{detail} body={raw}"
            logger.warning("resend send failed: %s", detail)
            return DeliveryResult(channel="email", attempted=True, status="failed", detail=detail)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("resend send failed: %s", exc)
            return DeliveryResult(channel="email", attempted=True, status="failed", detail=str(exc))
