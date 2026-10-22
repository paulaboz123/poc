"""
report.py - one HTML page with everything generated so far: status of each video, what the
checks found, the model review, a thumbnail of every scene, and links to the video, audio
and script. Also text lessons and the estimated spend.

Usage:  uv run report.py      -> data/output/report.html (double-click to open in the browser)
"""
from __future__ import annotations

import csv
import html
import json
import logging
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import config

log = logging.getLogger("report")

STATUS_COLOR = {"OK": "#15803d", "CHECK": "#b45309", "FAIL": "#b91c1c", "PLAN": "#475569", "?": "#64748b"}
CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1f2933;background:#f8fafc}
h1{margin:0 0 4px}h2{margin-top:32px}.muted{color:#64748b}
table{border-collapse:collapse;width:100%;background:#fff}
th,td{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left;vertical-align:top;font-size:14px}
th{background:#f1f5f9}.badge{color:#fff;border-radius:4px;padding:2px 8px;font-weight:600;font-size:12px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;margin:12px 0}
.thumbs img{width:160px;margin:4px 4px 0 0;border:1px solid #cbd5e1;border-radius:4px}
ul{margin:4px 0 0 18px;padding:0}a{color:#0f766e}
"""


def _read(path: Path) -> dict:
    """JSON file as a dict ({} if missing or broken)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _badge(status: str) -> str:
    return f'<span class="badge" style="background:{STATUS_COLOR.get(status, "#64748b")}">{status}</span>'


def _link(folder: Path, name: str, label: str) -> str:
    """Relative link to a file in an output folder, or '' if the file does not exist."""
    return f'<a href="{html.escape(folder.name)}/{name}">{label}</a>' if (folder / name).exists() else ""


def video_rows() -> list[dict]:
    """One dict per video output folder, newest first."""
    rows = []
    for sb_path in sorted(config.OUTPUT_DIR.glob("*/storyboard.json"), reverse=True):
        folder = sb_path.parent
        sb, info, q = _read(sb_path), _read(folder / "info.json"), _read(folder / "quality.json")
        status = q.get("status") or ("PLAN" if info.get("plan_only") or not (folder / "lesson_video.mp4").exists() else "?")
        rows.append({"folder": folder, "sb": sb, "info": info, "q": q, "status": status})
    return rows


def lesson_rows() -> list[dict]:
    """One dict per text-lesson output folder, newest first."""
    return [{"folder": p.parent, "data": _read(p)}
            for p in sorted(config.OUTPUT_DIR.glob("*/lesson.json"), reverse=True)]


def spend_summary() -> dict:
    """Estimated spend from cost_log.csv: today, all time, and per step."""
    out = {"today": 0.0, "total": 0.0, "by_step": defaultdict(float)}
    if not config.COST_LOG.exists():
        return out
    today = date.today().isoformat()
    with open(config.COST_LOG, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            cost = float(r.get("est_cost_usd") or 0)
            out["total"] += cost
            out["by_step"][r.get("step", "?")] += cost
            if (r.get("time") or "").startswith(today):
                out["today"] += cost
    return out


def build() -> Path:
    """Write data/output/report.html and return its path."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    videos, lessons, spend = video_rows(), lesson_rows(), spend_summary()
    e = html.escape
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Lesson report</title><style>{CSS}</style></head><body>",
             "<h1>Video lesson report</h1>",
             f"<p class='muted'>{datetime.now():%Y-%m-%d %H:%M} · {len(videos)} videos · {len(lessons)} text lessons · "
             f"estimated spend today ${spend['today']:.2f} (lock ${config.DAILY_BUDGET_USD:.0f}/day), "
             f"all time ${spend['total']:.2f}</p>"]

    counts = defaultdict(int)
    for v in videos:
        counts[v["status"]] += 1
    parts.append("<p>" + " ".join(f"{_badge(k)} {n}" for k, n in sorted(counts.items())) + "</p>")

    # overview table
    parts.append("<h2>Videos</h2><table><tr><th>Status</th><th>Title</th><th>Request</th><th>Model</th>"
                 "<th>Length</th><th>Review</th><th>Cost</th><th>Files</th></tr>")
    for v in videos:
        sb, info, q, f = v["sb"], v["info"], v["q"], v["folder"]
        req = sb.get("request") or {"topic": sb.get("topic"), "minutes": sb.get("minutes"), "level": sb.get("level")}
        req_txt = (f"rec. {req.get('recording')}" if req.get("recording") else e(str(req.get("topic", ""))[:50])) + \
                  f" · {req.get('minutes', '')} min · {req.get('level', '')}"
        dur = info.get("duration_s")
        length = f"{int(dur) // 60}:{int(dur) % 60:02d}" if dur else "-"
        rv = q.get("review")
        review = f"F{rv['faithful']} L{rv['level_fit']} C{rv['clarity']}" if rv else "-"
        files = " · ".join(x for x in [_link(f, "lesson_video.mp4", "video"), _link(f, "lesson_audio.mp3", "audio"),
                                        _link(f, "script.md", "script")] if x)
        parts.append(f"<tr><td>{_badge(v['status'])}</td><td><a href='#{e(f.name)}'>{e(sb.get('title', f.name))}</a></td>"
                     f"<td>{req_txt}</td><td class='muted'>{e(str(sb.get('model', '')).split('/')[-1][:40])}</td>"
                     f"<td>{length}</td><td>{review}</td><td>${info.get('model_cost_usd_est', 0):.3f}</td><td>{files}</td></tr>")
    parts.append("</table>")

    # details per video
    for v in videos:
        sb, q, f = v["sb"], v["q"], v["folder"]
        parts.append(f"<div class='card' id='{e(f.name)}'><h3>{_badge(v['status'])} {e(sb.get('title', ''))}</h3>"
                     f"<p class='muted'>{e(f.name)} · goal: {e(sb.get('goal', ''))}</p>")
        problems = q.get("fail", []) + q.get("issues", [])
        if q.get("review"):
            problems += [f"review, scene {i.get('scene')}: {i.get('problem')}" for i in q["review"].get("issues", [])]
        if problems:
            parts.append("<ul>" + "".join(f"<li>{e(str(p))}</li>" for p in problems) + "</ul>")
        elif not q:
            parts.append("<p class='muted'>Not checked yet: uv run quality.py --all</p>")
        thumbs = sorted((f / "scenes").glob("scene_*.png"))
        if thumbs:
            parts.append("<div class='thumbs'>" + "".join(
                f"<img src='{e(f.name)}/scenes/{t.name}' title='{e(sc.get('narration', '')[:300])}'>"
                for t, sc in zip(thumbs, sb.get("scenes", []))) + "</div>")
        parts.append("</div>")

    # text lessons
    if lessons:
        parts.append("<h2>Text lessons</h2><table><tr><th>Title</th><th>Topic</th><th>Words</th><th>Length ok</th>"
                     "<th>Invented sources</th><th>Model</th><th>File</th></tr>")
        for row in lessons:
            d, f = row["data"], row["folder"]
            st = d.get("stats", {})
            parts.append(f"<tr><td>{e(d.get('lesson', {}).get('title', ''))}</td><td>{e(str(d.get('topic', '')))}</td>"
                         f"<td>{st.get('words', '')}</td><td>{'yes' if st.get('length_ok') else 'no'}</td>"
                         f"<td>{e(str(st.get('invalid_citations', [])))}</td><td class='muted'>{e(str(d.get('model', '')).split('/')[-1])}</td>"
                         f"<td>{_link(f, 'lesson.md', 'lesson.md')}</td></tr>")
        parts.append("</table>")

    # spend per step
    if spend["by_step"]:
        parts.append("<h2>Estimated spend per step (all time)</h2><table><tr><th>Step</th><th>USD</th></tr>" +
                     "".join(f"<tr><td>{e(k)}</td><td>${val:.3f}</td></tr>"
                             for k, val in sorted(spend["by_step"].items(), key=lambda kv: -kv[1])) + "</table>")
    parts.append("</body></html>")
    out = config.OUTPUT_DIR / "report.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


if __name__ == "__main__":
    config.setup_logging()
    path = build()
    print(f"Report: {path}  (open it in the browser)")
    sys.exit(0)
