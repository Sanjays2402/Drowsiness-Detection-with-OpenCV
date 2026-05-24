"""HTML report rendering for ``drowsy analyze``.

Produces a single self-contained ``report.html`` from a finished
analyze run (ear.csv + events.jsonl + summary.json). No JavaScript
dependencies — the EAR chart is rendered as inline SVG so the file
works offline, prints cleanly, and can be emailed as one attachment.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# Visual constants — kept here so the chart is easy to retune without
# digging through f-string land below.
_CHART_W = 1160
_CHART_H = 320
_CHART_PAD_L = 56
_CHART_PAD_R = 24
_CHART_PAD_T = 24
_CHART_PAD_B = 36
_PLOT_W = _CHART_W - _CHART_PAD_L - _CHART_PAD_R
_PLOT_H = _CHART_H - _CHART_PAD_T - _CHART_PAD_B


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


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
        # Bucket is "drowsy" if any frame in it was drowsy — preserves
        # event visibility through downsampling.
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
            f'<rect width="{_CHART_W}" height="{_CHART_H}" fill="#0e0e12" rx="14"/>'
            f'<text x="{_CHART_W/2}" y="{_CHART_H/2}" fill="#8b8b94" '
            'text-anchor="middle" font-family="sans-serif" font-size="14">'
            "no EAR data</text></svg>"
        )

    sampled = _downsample(series)
    t_min = sampled[0][0]
    t_max = sampled[-1][0]
    ears = [s[1] for s in sampled if s[1] > 0]
    ear_lo = min(min(ears, default=0.0), threshold) - 0.02
    ear_hi = max(max(ears, default=threshold), threshold) + 0.02
    ear_lo = max(ear_lo, 0.0)

    points = list(_project(sampled, t_min, t_max, ear_lo, ear_hi))

    # Build line path. Break the line into segments when state flips so
    # we can color drowsy stretches red without re-projecting.
    segments: List[List[Tuple[float, float]]] = []
    current_state = None
    current: List[Tuple[float, float]] = []
    for x, y, st in points:
        if st != current_state and current:
            segments.append(current)
            current = [current[-1]]  # bridge to keep the line continuous
        current.append((x, y))
        current_state = st
    if current:
        segments.append(current)

    def _path(pts: List[Tuple[float, float]]) -> str:
        return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    # Y-axis ticks at fixed EAR values; X-axis ticks at 6 evenly spaced
    # times.
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
        'aria-label="Eye aspect ratio over time">'
    )
    parts.append(
        '<defs><linearGradient id="ear-fill" x1="0" x2="0" y1="0" y2="1">'
        '<stop offset="0%" stop-color="#ffc933" stop-opacity="0.28"/>'
        '<stop offset="100%" stop-color="#ffc933" stop-opacity="0"/>'
        "</linearGradient></defs>"
    )
    parts.append(
        f'<rect width="{_CHART_W}" height="{_CHART_H}" fill="#0e0e12" rx="14"/>'
    )

    # Gridlines
    for v in y_ticks:
        y = _CHART_PAD_T + (1.0 - (v - ear_lo) / max(ear_hi - ear_lo, 1e-6)) * _PLOT_H
        parts.append(
            f'<line x1="{_CHART_PAD_L}" x2="{_CHART_W - _CHART_PAD_R}" '
            f'y1="{y:.1f}" y2="{y:.1f}" stroke="rgba(255,255,255,0.05)"/>'
        )
        parts.append(
            f'<text x="{_CHART_PAD_L - 8}" y="{y + 4:.1f}" fill="#8b8b94" '
            'font-family="ui-sans-serif,system-ui" font-size="11" '
            f'text-anchor="end">{v:.2f}</text>'
        )

    # Threshold line
    parts.append(
        f'<line x1="{_CHART_PAD_L}" x2="{_CHART_W - _CHART_PAD_R}" '
        f'y1="{threshold_y:.1f}" y2="{threshold_y:.1f}" '
        'stroke="#ff6b78" stroke-dasharray="4 4" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{_CHART_W - _CHART_PAD_R - 6}" y="{threshold_y - 6:.1f}" '
        'fill="#ff6b78" font-family="ui-sans-serif,system-ui" font-size="11" '
        f'text-anchor="end">threshold {threshold:.2f}</text>'
    )

    # Area fill under the awake portions
    if points:
        first_x = points[0][0]
        last_x = points[-1][0]
        base_y = _CHART_PAD_T + _PLOT_H
        fill_pts = [(first_x, base_y)] + [(x, y) for x, y, _ in points] + [(last_x, base_y)]
        parts.append(
            f'<path d="{_path(fill_pts)} Z" fill="url(#ear-fill)" opacity="0.8"/>'
        )

    # Colored line segments
    for seg in segments:
        if len(seg) < 2:
            continue
        # Determine color from the segment's dominant state by looking
        # up the original points whose x matches the start. Simpler:
        # color drowsy segments red, others amber.
        end_x, end_y = seg[-1]
        # Find state by lookup in points (last entry's state).
        seg_state = "awake"
        for x, y, st in points:
            if abs(x - end_x) < 1e-3 and abs(y - end_y) < 1e-3:
                seg_state = st
                break
        color = "#ff6b78" if seg_state == "drowsy" else "#ffc933"
        parts.append(
            f'<path d="{_path(seg)}" fill="none" stroke="{color}" '
            'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # Event markers
    for ev in events:
        t = float(ev.get("t_seconds", 0.0))
        if t < t_min or t > t_max:
            continue
        x = (
            _CHART_PAD_L
            + (t - t_min) / max(t_max - t_min, 1e-6) * _PLOT_W
        )
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" '
            f'y1="{_CHART_PAD_T}" y2="{_CHART_PAD_T + _PLOT_H}" '
            'stroke="#ff6b78" stroke-width="1" stroke-dasharray="2 3" opacity="0.7"/>'
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{_CHART_PAD_T + 8}" r="4" '
            'fill="#ff6b78"/>'
        )

    # X-axis ticks
    for t in x_ticks:
        x = (
            _CHART_PAD_L
            + (t - t_min) / max(t_max - t_min, 1e-6) * _PLOT_W
        )
        parts.append(
            f'<text x="{x:.1f}" y="{_CHART_PAD_T + _PLOT_H + 18}" fill="#8b8b94" '
            'font-family="ui-sans-serif,system-ui" font-size="11" '
            f'text-anchor="middle">{_fmt_duration(t)}</text>'
        )

    # Axis baseline
    parts.append(
        f'<line x1="{_CHART_PAD_L}" x2="{_CHART_W - _CHART_PAD_R}" '
        f'y1="{_CHART_PAD_T + _PLOT_H:.1f}" y2="{_CHART_PAD_T + _PLOT_H:.1f}" '
        'stroke="rgba(255,255,255,0.12)"/>'
    )

    parts.append("</svg>")
    return "".join(parts)


def _events_table(events: Sequence[dict]) -> str:
    if not events:
        return (
            '<div class="empty">No drowsiness events detected in this video. \U0001f44d</div>'
        )
    rows = []
    for i, ev in enumerate(events, 1):
        t = float(ev.get("t_seconds", 0.0))
        ear = float(ev.get("ear", 0.0))
        frames = int(ev.get("closed_frames", 0))
        rows.append(
            f"<tr><td class=\"n\">{i}</td>"
            f"<td>{html.escape(_fmt_duration(t))}</td>"
            f"<td>{ear:.3f}</td>"
            f"<td>{frames}</td></tr>"
        )
    body = "".join(rows)
    return (
        '<table class="events"><thead>'
        '<tr><th>#</th><th>Time</th><th>EAR</th><th>Closed frames</th></tr>'
        f'</thead><tbody>{body}</tbody></table>'
    )


_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Drowsiness Report \u00b7 {source}</title>
<style>
:root {{
  --bg: #08080b;
  --surface: rgba(255,255,255,0.03);
  --border: rgba(255,255,255,0.08);
  --border-strong: rgba(255,255,255,0.14);
  --text: #ececf1;
  --muted: #8b8b94;
  --accent: #ffc933;
  --danger: #ff6b78;
  --radius: 14px;
}}
* {{ box-sizing: border-box; }}
html, body {{
  margin: 0; padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}}
body {{
  background:
    radial-gradient(900px 600px at 12% -10%, rgba(255,201,51,0.10), transparent 60%),
    radial-gradient(700px 500px at 92% 8%, rgba(255,107,120,0.08), transparent 60%),
    var(--bg);
  background-attachment: fixed;
  min-height: 100vh;
  padding: 48px 24px 64px;
}}
.wrap {{ max-width: 1200px; margin: 0 auto; }}
header {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; margin-bottom: 32px; flex-wrap: wrap; }}
.brand {{ font-size: 12px; letter-spacing: 0.08em; color: var(--accent); text-transform: uppercase; font-weight: 600; margin-bottom: 6px; }}
h1 {{
  font-size: clamp(26px, 3.6vw, 38px);
  letter-spacing: -0.025em;
  margin: 0;
  background: linear-gradient(180deg, #fff 0%, #c8c8d0 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.source {{ color: var(--muted); font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size: 13px; margin-top: 6px; word-break: break-all; }}
.meta {{ text-align: right; color: var(--muted); font-size: 13px; line-height: 1.6; }}
.stats {{
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
  margin-bottom: 32px;
}}
@media (max-width: 760px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} }}
.stat {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  transition: border-color 180ms ease;
}}
.stat:hover {{ border-color: var(--border-strong); }}
.stat .label {{ font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }}
.stat .value {{ font-size: 28px; font-weight: 700; letter-spacing: -0.02em; }}
.stat .value.danger {{ color: var(--danger); }}
.stat .value.accent {{ color: var(--accent); }}
.stat .sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px;
  margin-bottom: 22px;
}}
.card h2 {{ margin: 0 0 16px; font-size: 14px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); font-weight: 600; }}
.chart-wrap {{ width: 100%; overflow-x: auto; }}
.chart-wrap svg {{ display: block; width: 100%; height: auto; min-width: 760px; }}
.events {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.events th, .events td {{ padding: 10px 12px; text-align: left; }}
.events th {{ font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); }}
.events tbody tr {{ border-bottom: 1px solid var(--border); }}
.events tbody tr:last-child {{ border-bottom: none; }}
.events td.n {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
.empty {{ color: var(--muted); padding: 14px 0; font-size: 14px; }}
footer {{ margin-top: 32px; text-align: center; color: var(--muted); font-size: 12px; }}
footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div class="brand">Drowsy \u00b7 Report</div>
      <h1>Drowsiness analysis</h1>
      <div class="source">{source}</div>
    </div>
    <div class="meta">
      Generated {generated}<br/>
      drowsy v{version} \u00b7 EAR threshold {threshold:.2f}
    </div>
  </header>

  <div class="stats">
    <div class="stat">
      <div class="label">Frames</div>
      <div class="value">{frames}</div>
      <div class="sub">{duration} \u00b7 {fps:.1f} fps</div>
    </div>
    <div class="stat">
      <div class="label">Drowsy events</div>
      <div class="value {events_color}">{event_count}</div>
      <div class="sub">{event_rate}</div>
    </div>
    <div class="stat">
      <div class="label">Closed-eye frames</div>
      <div class="value">{closed_frames}</div>
      <div class="sub">{closed_pct:.1f}% of total</div>
    </div>
    <div class="stat">
      <div class="label">Mean EAR</div>
      <div class="value accent">{mean_ear:.3f}</div>
      <div class="sub">min {min_ear:.3f} \u00b7 max {max_ear:.3f}</div>
    </div>
  </div>

  <div class="card">
    <h2>Eye aspect ratio over time</h2>
    <div class="chart-wrap">{chart}</div>
  </div>

  <div class="card">
    <h2>Events ({event_count})</h2>
    {events_table}
  </div>

  <footer>
    Self-contained report \u00b7 no JavaScript \u00b7
    <a href="https://github.com/Sanjays2402/Drowsiness-Detection-with-OpenCV">source on GitHub</a>
  </footer>
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
        if duration_s > 0 else "\u2014"
    )

    from datetime import datetime
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    chart = _chart_svg(series, threshold, events)

    body = _TEMPLATE.format(
        source=html.escape(str(summary.get("source", ""))),
        generated=html.escape(generated),
        version=html.escape(version),
        threshold=threshold,
        frames=f"{frames:,}",
        duration=_fmt_duration(duration_s) if duration_s else "\u2014",
        fps=fps,
        event_count=len(events),
        events_color="danger" if events else "",
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
