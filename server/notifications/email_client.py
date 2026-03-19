from __future__ import annotations

import logging
import smtplib

from email.message import EmailMessage

from .config import NotificationConfig
from .models import DeliveryResult


logger = logging.getLogger(__name__)


class EmailClient:
    """SMTP email sender."""

    def __init__(self, config: NotificationConfig):
        self._cfg = config

    def send(self, msg: EmailMessage) -> DeliveryResult:
        if not (self._cfg.smtp_host and self._cfg.email_from and self._cfg.caregiver_email):
            return DeliveryResult(channel="email", attempted=False, status="skipped_missing_config", detail="")
        try:
            with smtplib.SMTP(self._cfg.smtp_host, self._cfg.smtp_port, timeout=self._cfg.http_timeout_s) as smtp:
                smtp.ehlo()
                try:
                    smtp.starttls()
                    smtp.ehlo()
                except smtplib.SMTPException:
                    pass
                if self._cfg.smtp_username:
                    smtp.login(self._cfg.smtp_username, self._cfg.smtp_password)
                smtp.send_message(msg)
            return DeliveryResult(channel="email", attempted=True, status="sent", detail="")
        except (smtplib.SMTPException, OSError) as exc:
            logger.warning("smtp send failed: %s", exc)
            return DeliveryResult(channel="email", attempted=True, status="failed", detail=str(exc))
