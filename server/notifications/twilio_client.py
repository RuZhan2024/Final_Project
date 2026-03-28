from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from typing import Optional

from .config import NotificationConfig
from .models import DeliveryResult


logger = logging.getLogger(__name__)


class TwilioClient:
    """Minimal Twilio REST wrapper using the standard library."""

    def __init__(self, config: NotificationConfig):
        self._cfg = config

    def _auth_header(self) -> str:
        token = f"{self._cfg.twilio_account_sid}:{self._cfg.twilio_auth_token}".encode("utf-8")
        return f"Basic {base64.b64encode(token).decode('ascii')}"

    def _post(self, url: str, data: dict[str, str]) -> tuple[bool, str]:
        body = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", self._auth_header())
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.http_timeout_s) as resp:
                payload = resp.read().decode("utf-8", errors="replace")
                return True, payload[:500]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.warning("twilio request failed status=%s detail=%s", exc.code, detail[:500])
            return False, detail[:500]
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("twilio request error: %s", exc)
            return False, str(exc)

    def call(self, *, to_phone: str, message: str) -> DeliveryResult:
        if not (self._cfg.twilio_account_sid and self._cfg.twilio_auth_token and self._cfg.twilio_from_phone and to_phone):
            return DeliveryResult(channel="phone", attempted=False, status="skipped_missing_config", detail="")
        twiml = f"<Response><Say>{message}</Say></Response>"
        ok, detail = self._post(
            f"https://api.twilio.com/2010-04-01/Accounts/{self._cfg.twilio_account_sid}/Calls.json",
            {"To": to_phone, "From": self._cfg.twilio_from_phone, "Twiml": twiml},
        )
        return DeliveryResult(channel="phone", attempted=True, status="sent" if ok else "failed", detail=detail)

    def sms(self, *, to_phone: str, message: str) -> DeliveryResult:
        if not (self._cfg.twilio_account_sid and self._cfg.twilio_auth_token and self._cfg.twilio_from_phone and to_phone):
            return DeliveryResult(channel="sms", attempted=False, status="skipped_missing_config", detail="")
        ok, detail = self._post(
            f"https://api.twilio.com/2010-04-01/Accounts/{self._cfg.twilio_account_sid}/Messages.json",
            {"To": to_phone, "From": self._cfg.twilio_from_phone, "Body": message},
        )
        return DeliveryResult(channel="sms", attempted=True, status="sent" if ok else "failed", detail=detail)
