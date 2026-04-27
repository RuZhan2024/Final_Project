from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.error
import urllib.request

from .config import NotificationConfig
from .models import SafeGuardEvent, TierDecision


logger = logging.getLogger(__name__)


def _fallback_report(event: SafeGuardEvent, decision: TierDecision) -> str:
    risk = "high" if decision.tier.value == "tier1_high_confidence_fall" else "moderate"
    return (
        f"Summary: A {risk}-risk fall-like event was detected at {event.location}. "
        f"The model reported triage_state={event.triage_state}, probability={event.probability:.3f}, "
        f"uncertainty={event.uncertainty:.3f}, and margin={event.margin:.3f}. "
        f"Recommended action: {decision.recommendation}"
    )


def _extract_output_text(payload: dict) -> str:
    txt = payload.get("output_text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                out = content.get("text", "").strip()
                if out:
                    return out
    return ""


def _build_prompt(event: SafeGuardEvent, decision: TierDecision) -> str:
    return "\n".join(
        [
            "You are generating a caregiver-facing fall event analysis email section.",
            "Write in plain English suitable for a non-technical caregiver.",
            "Use exactly these section headings:",
            "What happened",
            "Why the system is concerned",
            "What to do now",
            "Under each heading, write 1 to 3 short sentences.",
            "Be calm, specific, and practical.",
            "Do not mention internal implementation details, model names, or unsupported medical claims.",
            "Do not speculate beyond the provided data.",
            "If uncertainty is elevated, say the event may need confirmation while still advising prompt checking.",
            "",
            f"event_id: {event.event_id}",
            f"location: {event.location}",
            f"timestamp: {event.timestamp.isoformat()}",
            f"triage_state: {event.triage_state}",
            f"probability: {event.probability:.4f}",
            f"uncertainty: {event.uncertainty:.4f}",
            f"threshold: {event.threshold:.4f}",
            f"margin: {event.margin:.4f}",
            f"alert_tier: {decision.tier.value}",
            f"decision_reason: {decision.reason}",
            f"recommendation: {decision.recommendation}",
            f"dataset_code: {event.dataset_code}",
            f"model_code: {event.model_code}",
            f"op_code: {event.op_code}",
        ]
    )


def _extract_gemini_text(payload: dict) -> str:
    for candidate in payload.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    return ""


def _generate_openai_report(prompt: str, cfg: NotificationConfig) -> str:
    body = json.dumps(
        {
            "model": cfg.openai_model,
            "input": prompt,
            "max_output_tokens": 220,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {cfg.openai_api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=cfg.openai_timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return _extract_output_text(payload)


def _generate_gemini_report(prompt: str, cfg: NotificationConfig) -> str:
    model = urllib.parse.quote(cfg.gemini_model, safe="")
    key = urllib.parse.quote(cfg.gemini_api_key, safe="")
    body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 220,
                "temperature": 0.3,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=cfg.openai_timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return _extract_gemini_text(payload)


def generate_event_ai_report(event: SafeGuardEvent, decision: TierDecision, cfg: NotificationConfig) -> str:
    if not cfg.ai_reports_enabled:
        return _fallback_report(event, decision)
    prompt = _build_prompt(event, decision)
    provider = str(cfg.ai_provider or "openai").strip().lower()
    try:
        if provider == "gemini":
            if not cfg.gemini_api_key or not cfg.gemini_model:
                return _fallback_report(event, decision)
            out = _generate_gemini_report(prompt, cfg)
        else:
            if not cfg.openai_api_key or not cfg.openai_model:
                return _fallback_report(event, decision)
            out = _generate_openai_report(prompt, cfg)
        return out or _fallback_report(event, decision)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("ai report generation failed provider=%s event_id=%s: %s", provider, event.event_id, exc)
        return _fallback_report(event, decision)
