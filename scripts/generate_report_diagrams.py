#!/usr/bin/env python3
"""Generate simple report-ready SVG diagrams for the final report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple


SVG_HEADER = """<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/>
  </marker>
</defs>
<rect width="100%" height="100%" fill="#ffffff"/>
"""


def _svg_text(x: int, y: int, text: str, size: int = 18, weight: str = "600", anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="#0f172a">{text}</text>'
    )


def _svg_box(x: int, y: int, w: int, h: int, label: str, fill: str) -> str:
    cx = x + w // 2
    cy = y + h // 2 + 6
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" rx="18" ry="18" width="{w}" height="{h}" fill="{fill}" stroke="#334155" stroke-width="2"/>',
            _svg_text(cx, cy, label, size=20, weight="600"),
        ]
    )


def _svg_group_label(x: int, y: int, w: int, h: int, label: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" rx="22" ry="22" width="{w}" height="{h}" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="10 8"/>',
            _svg_text(x + 18, y + 32, label, size=18, weight="700", anchor="start"),
        ]
    )


def _svg_arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334155" stroke-width="3" marker-end="url(#arrow)"/>'


def _wrap_svg(width: int, height: int, body: Iterable[str]) -> str:
    return SVG_HEADER.format(w=width, h=height) + "\n".join(body) + "\n</svg>\n"


def build_system_architecture_svg() -> str:
    width, height = 1600, 900
    body = [
        _svg_text(width // 2, 50, "Figure 1. System Architecture and Decision Path", size=28, weight="700"),
        _svg_group_label(60, 110, 620, 690, "Frontend / Client"),
        _svg_group_label(900, 110, 640, 690, "Backend / Runtime"),
    ]

    boxes = {
        "video": (130, 190, 220, 90, "Camera / Replay Video", "#dbeafe"),
        "pose": (430, 190, 200, 90, "Pose Extraction", "#dbeafe"),
        "window": (270, 370, 220, 90, "Temporal Window Builder", "#dbeafe"),
        "ui": (270, 560, 220, 90, "Monitor UI State", "#dbeafe"),
        "runtime": (980, 190, 210, 90, "Inference Runtime", "#dcfce7"),
        "model": (1260, 190, 190, 90, "TCN / GCN Output", "#dcfce7"),
        "policy": (1090, 370, 250, 90, "Operating Point + Policy", "#dcfce7"),
        "persist": (980, 560, 220, 90, "Event Persistence", "#dcfce7"),
        "notify": (1280, 560, 190, 90, "Notifications / History", "#dcfce7"),
    }
    for x, y, w, h, label, fill in boxes.values():
        body.append(_svg_box(x, y, w, h, label, fill))

    body.extend(
        [
            _svg_arrow(350, 235, 430, 235),
            _svg_arrow(530, 280, 530, 330),
            _svg_arrow(490, 415, 980, 235),
            _svg_arrow(1190, 235, 1260, 235),
            _svg_arrow(1355, 280, 1355, 330),
            _svg_arrow(1190, 280, 1190, 370),
            _svg_arrow(1215, 460, 1215, 530),
            _svg_arrow(1200, 605, 1280, 605),
            _svg_arrow(1090, 415, 490, 605),
        ]
    )

    body.extend(
        [
            _svg_text(790, 210, "window payload", size=16, weight="500"),
            _svg_text(1360, 335, "score stream", size=16, weight="500"),
            _svg_text(1215, 515, "events", size=16, weight="500"),
            _svg_text(805, 610, "final monitor state", size=16, weight="500"),
        ]
    )
    return _wrap_svg(width, height, body)


def build_alert_policy_svg() -> str:
    width, height = 1500, 520
    body = [
        _svg_text(width // 2, 50, "Figure 2. Alert Policy Decision Flow", size=28, weight="700"),
    ]

    steps: Tuple[Tuple[int, str, str], ...] = (
        (90, "Window-level model score", "#e0f2fe"),
        (360, "Validation-fitted operating point", "#fef3c7"),
        (660, "Temporal smoothing / EMA", "#dcfce7"),
        (950, "k-of-n alert logic", "#fde68a"),
        (1180, "Cooldown / confirmation", "#fecaca"),
    )

    for x, label, fill in steps:
        body.append(_svg_box(x, 180, 220, 100, label, fill))

    body.append(_svg_box(1280, 360, 160, 90, "Final alert state", "#ddd6fe"))

    body.extend(
        [
            _svg_arrow(310, 230, 360, 230),
            _svg_arrow(580, 230, 660, 230),
            _svg_arrow(880, 230, 950, 230),
            _svg_arrow(1170, 230, 1180, 230),
            _svg_arrow(1290, 280, 1360, 350),
        ]
    )

    body.extend(
        [
            _svg_text(200, 330, "Raw probability is not yet an operational decision.", size=17, weight="500"),
            _svg_text(785, 330, "The deployed monitor consumes a fitted policy profile, not a single ad hoc threshold.", size=17, weight="500"),
        ]
    )
    return _wrap_svg(width, height, body)


def main() -> None:
    out_dir = Path("artifacts/figures/report")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "system_architecture_diagram.svg").write_text(build_system_architecture_svg(), encoding="utf-8")
    (out_dir / "alert_policy_flow.svg").write_text(build_alert_policy_svg(), encoding="utf-8")

    print("[ok] generated report diagrams")
    print(f"[info] output_dir={out_dir}")


if __name__ == "__main__":
    main()
