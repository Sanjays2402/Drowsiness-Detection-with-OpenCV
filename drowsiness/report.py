"""HTML report rendering for ``drowsy analyze``.

Produces a single self-contained ``report.html`` from a finished
analyze run (ear.csv + events.jsonl + summary.json). No JavaScript
dependencies \u2014 the EAR chart is rendered as inline SVG so the file
works offline, prints cleanly, and can be emailed as one attachment.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# Visual constants \u2014 kept here so the chart is easy to retune without
# digging through f-string land below.
_CHART_W = 1160
_CHART_H = 280
_CHART_PAD_L = 48
_CHART_PAD_R = 16
_CHART_PAD_T = 18
_CHART_PAD_B = 28
_PLOT_W = _CHART_W - _CHART_PAD_L - _CHART_PAD_R
_PLOT_H = _CHART_H - _CHART_PAD_T - _CHART_PAD_B


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _read_ear_series(csv_path: Path) -> List[Tuple[float, float, str]]:
    import csv

    out: List[Tuple[float, float, str]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["t_seconds"])
                ear = float(row["ear"])
            except (TypeError, ValueError):
                continue
            out.append((t, ear, row.get("state", "awake")))
    return out


def _downsample(series: Sequence[Tuple[float, float, str]], max_points: int = 1200
                ) -> List[Tuple[float, float, str]]:
    """Bucket-average to keep the inline SVG small on long videos."""
    if len(series) <= max_points:
        return list(series)
    step = len(series) / max_points
    out: List[Tuple[float, float, str]] = []
    for i in range(max_points):
        a = int(i * step)
        b = max(a + 1, int((i + 1) * step))
        chunk = series[a:b]
        if not chunk:
            continue
        t = sum(c[0] for c in chunk) / len(chunk)
        ear = sum(c[1] for c in chunk) / len(chunk)
        state = "drowsy" if any(c[2] == "drowsy" for c in chunk) else "awake"
        out.append((t, ear, state))
    return out


def _project(
    series: Sequence[Tuple[float, float, str]],
    t_min: float,
    t_max: float,
    ear_min: float,
    ear_max: float,
) -> Iterable[Tuple[float, float, str]]:
    t_span = max(t_max - t_min, 1e-6)
    e_span = max(ear_max - ear_min, 1e-6)
    for t, ear, state in series:
        x = _CHART_PAD_L + (t - t_min) / t_span * _PLOT_W
        y = _CHART_PAD_T + (1.0 - (ear - ear_min) / e_span) * _PLOT_H
        yield x, y, state


def _chart_svg(
    series: Sequence[Tuple[float, float, str]],
    threshold: float,
    events: Sequence[dict],
) -> str:
    if not series:
        return (
            f'<svg viewBox="0 0 {_CHART_W} {_CHART_H}" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="{_CHART_W}" height="{_CHART_H}" fill="#0a0a0a"/>'
            f'<text x="{_CHART_W/2}" y="{_CHART_H/2}" fill="#666" '
            'text-anchor="middle" font-family="ui-monospace,monospace" font-size="12">'
            "no ear data</text></svg>"
        )

    sampled = _downsample(series)
    t_min = sampled[0][0]
    t_max = sampled[-1][0]
    ears = [s[1] for s in sampled if s[1] > 0]
    ear_lo = min(min(ears, default=0.0), threshold) - 0.02
    ear_hi = max(max(ears, default=threshold), threshold) + 0.02
    ear_lo = max(ear_lo, 0.0)

    points = list(_project(sampled, t_min, t_max, ear_lo, ear_hi))

    # Segment the line on state flips so drowsy stretches render red.
    segments: List[Tuple[str, List[Tuple[float, float]]]] = []
    current_state = None
    current: List[Tuple[float, float]] = []
    for x, y, st in points:
        if st != current_state and current:
            segments.append((current_state, current))
            current = [current[-1]]
        current.append((x, y))
        current_state = st
    if current and current_state is not None:
        segments.append((current_state, current))

    def _path(pts: List[Tuple[float, float]]) -> str:
        return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    y_ticks = [v for v in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40) if ear_lo <= v <= ear_hi]
    x_ticks = [t_min + (t_max - t_min) * i / 5 for i in range(6)]

    threshold_y = (
        _CHART_PAD_T
        + (1.0 - (threshold - ear_lo) / max(ear_hi - ear_lo, 1e-6)) * _PLOT_H
    )

    parts: List[str] = []
    parts.append(
        f'<svg viewBox="0 0 {_CHART_W} {_CHART_H}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="EAR over time">'
    )
    parts.append(f'<rect width="{_CHART_W}" height="{_CHART_H}" fill="#0a0a0a"/>')

    # Gridlines and y-axis labels.
    for v in y_ticks:
        y = _CHART_PAD_T + (1.0 - (v - ear_lo) / max(ear_hi - ear_lo, 1e-6)) * _PLOT_H
        parts.append(
            f'<line x1="{_CHART_PAD_L}" x2="{_CHART_W - _CHART_PAD_R}" '
            f'y1="{y:.1f}" y2="{y:.1f}" stroke="#1a1a1a"/>'
        )
        parts.append(
            f'<text x="{_CHART_PAD_L - 6}" y="{y + 3.5:.1f}" fill="#666" '
            'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
            f'font-size="10" text-anchor="end">{v:.2f}</text>'
        )

    # Threshold line.
    parts.append(
        f'<line x1="{_CHART_PAD_L}" x2="{_CHART_W - _CHART_PAD_R}" '
        f'y1="{threshold_y:.1f}" y2="{threshold_y:.1f}" '
        'stroke="#c44" stroke-dasharray="3 3" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{_CHART_W - _CHART_PAD_R - 4}" y="{threshold_y - 4:.1f}" '
        'fill="#c44" font-family="ui-monospace,monospace" font-size="10" '
        f'text-anchor="end">thr={threshold:.2f}</text>'
    )

    # Line segments.
    for st, seg in segments:
        if len(seg) < 2:
            continue
        color = "#c44" if st == "drowsy" else "#c8c8c8"
        parts.append(
            f'<path d="{_path(seg)}" fill="none" stroke="{color}" '
            'stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # Event markers \u2014 vertical line + small triangle at top.
    for ev in events:
        t = float(ev.get("t_seconds", 0.0))
        if t < t_min or t > t_max:
            continue
        x = _CHART_PAD_L + (t - t_min) / max(t_max - t_min, 1e-6) * _PLOT_W
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" '
            f'y1="{_CHART_PAD_T}" y2="{_CHART_PAD_T + _PLOT_H}" '
            'stroke="#c44" stroke-width="0.6" opacity="0.5"/>'
        )
        parts.append(
            f'<polygon points="{x-3:.1f},{_CHART_PAD_T} '
            f'{x+3:.1f},{_CHART_PAD_T} {x:.1f},{_CHART_PAD_T + 5}" fill="#c44"/>'
        )

    # X-axis tick labels.
    for t in x_ticks:
        x = _CHART_PAD_L + (t - t_min) / max(t_max - t_min, 1e-6) * _PLOT_W
        parts.append(
            f'<text x="{x:.1f}" y="{_CHART_PAD_T + _PLOT_H + 14}" fill="#666" '
            'font-family="ui-monospace,monospace" font-size="10" '
            f'text-anchor="middle">{_fmt_duration(t)}</text>'
        )

    # Frame border (functional, not decorative).
    parts.append(
        f'<rect x="{_CHART_PAD_L}" y="{_CHART_PAD_T}" '
        f'width="{_PLOT_W}" height="{_PLOT_H}" fill="none" stroke="#2a2a2a"/>'
    )

    parts.append("</svg>")
    return "".join(parts)


def _events_table(events: Sequence[dict]) -> str:
    if not events:
        return '<div class="empty">no drowsy events</div>'
    rows = []
    for i, ev in enumerate(events, 1):
        t = float(ev.get("t_seconds", 0.0))
        ear = float(ev.get("ear", 0.0))
        frames = int(ev.get("closed_frames", 0))
        frame_no = int(ev.get("frame", 0))
        rows.append(
            f"<tr><td>{i}</td>"
            f"<td>{_fmt_duration(t)}</td>"
            f"<td>{frame_no}</td>"
            f"<td>{ear:.3f}</td>"
            f"<td>{frames}</td></tr>"
        )
    body = "".join(rows)
    return (
        '<table class="events"><thead>'
        '<tr><th>#</th><th>t</th><th>frame</th><th>ear</th><th>closed</th></tr>'
        f'</thead><tbody>{body}</tbody></table>'
    )


_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>drowsy report :: {source}</title>
<style>
* {{ box-sizing: border-box; }}
html, body {{
  margin: 0;
  background: #050505;
  color: #d4d4d4;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 13px;
  line-height: 1.5;
}}
body {{ padding: 24px; }}
.wrap {{ max-width: 1200px; margin: 0 auto; }}
.head {{
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 16px; flex-wrap: wrap;
  padding-bottom: 10px;
  border-bottom: 1px solid #1f1f1f;
  margin-bottom: 18px;
}}
.head h1 {{
  margin: 0;
  font-size: 13px;
  font-weight: 600;
  color: #d4d4d4;
  letter-spacing: 0;
}}
.head h1::before {{ content: "$ "; color: #555; }}
.head .right {{ color: #666; font-size: 12px; }}
.kv {{
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 2px 16px;
  margin-bottom: 18px;
  font-size: 12px;
}}
.kv dt {{ color: #666; }}
.kv dd {{ margin: 0; color: #d4d4d4; }}
.kv dd.flag-ok {{ color: #6a9955; }}
.kv dd.flag-warn {{ color: #d19a66; }}
.kv dd.flag-bad {{ color: #c44; }}
section {{ margin-bottom: 22px; }}
section h2 {{
  margin: 0 0 8px;
  font-size: 12px;
  font-weight: 600;
  color: #888;
}}
section h2::before {{ content: "# "; color: #444; }}
.chart-wrap svg {{ display: block; width: 100%; height: auto; min-width: 760px; }}
.events {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}}
.events th, .events td {{
  padding: 5px 12px 5px 0;
  text-align: left;
  font-weight: normal;
}}
.events th {{ color: #666; border-bottom: 1px solid #1f1f1f; }}
.events td {{ color: #d4d4d4; }}
.events td:first-child {{ color: #555; width: 32px; }}
.events tbody tr:hover {{ background: #0c0c0c; }}
.empty {{ color: #555; font-size: 12px; padding: 4px 0; }}
.legend {{ color: #666; font-size: 11px; margin-top: 6px; }}
.legend span {{ margin-right: 14px; }}
.legend i {{
  display: inline-block; width: 14px; height: 2px;
  vertical-align: middle; margin-right: 4px;
}}
.foot {{
  margin-top: 24px;
  padding-top: 10px;
  border-top: 1px solid #1f1f1f;
  color: #555;
  font-size: 11px;
}}
.foot a {{ color: #888; text-decoration: none; }}
.foot a:hover {{ color: #d4d4d4; text-decoration: underline; }}
@media print {{
  body {{ background: #fff; color: #000; }}
  .head, section h2, .kv dd, .events td {{ color: #000; }}
}}
</style>
</head>
<body>
<div class="wrap">

<div class="head">
  <h1>drowsy analyze {source}</h1>
  <div class="right">{generated} \u00b7 v{version}</div>
</div>

<dl class="kv">
  <dt>source</dt><dd>{source}</dd>
  <dt>frames</dt><dd>{frames} ({duration} @ {fps:.1f}fps)</dd>
  <dt>ear threshold</dt><dd>{threshold:.2f}</dd>
  <dt>events</dt><dd class="{events_class}">{event_count} ({event_rate})</dd>
  <dt>closed frames</dt><dd>{closed_frames} ({closed_pct:.1f}%)</dd>
  <dt>ear stats</dt><dd>mean={mean_ear:.3f} min={min_ear:.3f} max={max_ear:.3f}</dd>
</dl>

<section>
  <h2>ear over time</h2>
  <div class="chart-wrap">{chart}</div>
  <div class="legend">
    <span><i style="background:#c8c8c8"></i>awake</span>
    <span><i style="background:#c44"></i>drowsy</span>
    <span><i style="background:#c44;border-top:1px dashed;height:0"></i>threshold</span>
    <span>\u25bc event</span>
  </div>
</section>

<section>
  <h2>events ({event_count})</h2>
  {events_table}
</section>

<div class="foot">
  <a href="https://github.com/Sanjays2402/Drowsiness-Detection-with-OpenCV">github.com/Sanjays2402/Drowsiness-Detection-with-OpenCV</a>
</div>

</div>
</body>
</html>
"""


def render_report(report_dir: Path, version: str = "0.0.0") -> Path:
    """Render ``report.html`` from ``ear.csv`` + ``summary.json`` + events.

    Returns the path to the written HTML file.
    """
    report_dir = Path(report_dir)
    summary_path = report_dir / "summary.json"
    csv_path = report_dir / "ear.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {report_dir}")
    summary = json.loads(summary_path.read_text())

    series = _read_ear_series(csv_path) if csv_path.exists() else []
    ears = [e for _, e, _ in series if e > 0]
    mean_ear = sum(ears) / len(ears) if ears else 0.0
    min_ear = min(ears) if ears else 0.0
    max_ear = max(ears) if ears else 0.0

    threshold = float(summary.get("ear_threshold", 0.25))
    events = summary.get("events", [])
    frames = int(summary.get("frames", 0))
    fps = float(summary.get("input_fps", 0.0) or 0.0)
    duration_s = frames / fps if fps else 0.0
    closed_frames = int(summary.get("closed_frames", 0))
    closed_pct = (closed_frames / frames * 100.0) if frames else 0.0

    event_rate = (
        f"{len(events) / (duration_s / 60.0):.2f}/min"
        if duration_s > 0 else "n/a"
    )
    events_class = "flag-ok" if not events else "flag-bad"

    from datetime import datetime
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    chart = _chart_svg(series, threshold, events)

    body = _TEMPLATE.format(
        source=html.escape(str(summary.get("source", ""))),
        generated=html.escape(generated),
        version=html.escape(version),
        threshold=threshold,
        frames=f"{frames:,}",
        duration=_fmt_duration(duration_s) if duration_s else "0s",
        fps=fps,
        event_count=len(events),
        events_class=events_class,
        event_rate=html.escape(event_rate),
        closed_frames=f"{closed_frames:,}",
        closed_pct=closed_pct,
        mean_ear=mean_ear,
        min_ear=min_ear,
        max_ear=max_ear,
        chart=chart,
        events_table=_events_table(events),
    )

    out_path = report_dir / "report.html"
    out_path.write_text(body, encoding="utf-8")
    return out_path


__all__ = ["render_report"]
