"""
batch.py - makes many lessons in one go from a list, checks each one, and writes the report.
New lessons = new rows in requests.csv; nothing else to change.

requests.csv (copy requests.example.csv), one row per lesson:
  id         your number for the row
  type       video or lesson (text)
  topic      what to teach (searched in all recordings); can stay empty if recording is set
  recording  Excel id(s) of the recording(s) to use, e.g. 13 or 13,14 (videos only; optional)
  minutes    target length
  level      beginner, intermediate or expert
  model      optional: another model id for this row (e.g. Opus for a comparison)
  run        yes / no

Each row runs on its own: a failure is recorded and the batch goes on. Only the spending
lock stops the whole batch.

Usage:
  uv run batch.py                    # every row with run=yes
  uv run batch.py --plan-only        # storyboards only: fast and cheap, read the scripts first
  uv run batch.py --review           # plus the model review of every video (~$0.01 each)
  uv run batch.py --only 3,5         # just these ids
  uv run batch.py --file other.csv
Output: the usual folders in data/output, batch_<time>.csv (one line per row) and report.html.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import lesson
import llm
import make_video
import quality
import report

log = logging.getLogger("batch")

FIELDS = ["id", "type", "topic", "recording", "minutes", "level", "status", "folder",
          "duration_s", "cost_usd_est", "seconds", "problems"]


def read_requests(path: Path) -> list[dict]:
    """Rows of requests.csv (Excel-saved files with ';' work too), with defaults filled in."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Copy requests.example.csv to {path.name} and edit it.")
    text = path.read_text(encoding="utf-8-sig")
    first = text.splitlines()[0] if text else ""
    rows = csv.DictReader(text.splitlines(), delimiter=";" if first.count(";") > first.count(",") else ",")
    out = []
    for n, r in enumerate(rows, 1):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k}
        if not (r.get("topic") or r.get("recording")):
            continue
        r.setdefault("id", str(n))
        r["id"] = r.get("id") or str(n)
        r["type"] = (r.get("type") or "video").lower()
        r["minutes"] = r.get("minutes") or "2"
        r["level"] = (r.get("level") or "beginner").lower()
        r["run"] = (r.get("run") or "yes").lower()
        out.append(r)
    return out


def run_one(req: dict, plan_only: bool, review: bool) -> dict:
    """Make and check one request. Returns a result row (never raises, except the spending lock)."""
    t0, cost0 = time.perf_counter(), llm.spent_this_run()
    row = {k: req.get(k, "") for k in ("id", "type", "topic", "recording", "minutes", "level")}
    try:
        minutes = float(req["minutes"])
        if req["level"] not in lesson.LEVELS:
            raise ValueError(f"level must be one of {list(lesson.LEVELS)}")
        model = req.get("model") or None
        if req["type"] == "lesson":
            rec = lesson.make_lesson(req["topic"], int(round(minutes)), req["level"], model=model)
            st = rec["stats"]
            problems = ([f"length {st['words']} vs {st['target_words']} words"] if not st["length_ok"] else []) + \
                       ([f"invented citations {st['invalid_citations']}"] if st["invalid_citations"] else [])
            row.update(status="CHECK" if problems else "OK", folder=rec["path"], problems="; ".join(problems))
        elif req["type"] == "video":
            out_dir, info = make_video.make(req.get("topic") or None, req.get("recording") or None, minutes,
                                            req["level"], model, plan_only)
            row.update(folder=str(out_dir), duration_s=info.get("duration_s", ""))
            if plan_only:
                row.update(status="PLAN", problems="; ".join(info.get("storyboard_stats", {}).get("warnings", [])))
            else:
                q = quality.check(out_dir, review)
                row.update(status=q["status"], problems="; ".join(q["fail"] + q["issues"])[:500])
        else:
            raise ValueError(f"type must be video or lesson, got {req['type']!r}")
    except llm.BudgetExceeded:
        raise
    except lesson.NotEnoughMaterial as e:
        row.update(status="FAIL", problems=f"not covered: {e}"[:500])
    except (RuntimeError, ValueError, OSError) as e:
        row.update(status="FAIL", problems=str(e)[:500])
        log.error("Row %s failed: %s", row["id"], e)
    row["cost_usd_est"] = round(llm.spent_this_run() - cost0, 4)
    row["seconds"] = round(time.perf_counter() - t0)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=config.ROOT / "requests.csv")
    ap.add_argument("--only", help="comma-separated ids to run (ignores the run column)")
    ap.add_argument("--plan-only", action="store_true", help="storyboards only, no voice/video")
    ap.add_argument("--review", action="store_true", help="model review of every video (~$0.01 each)")
    args = ap.parse_args()
    config.setup_logging()
    try:
        requests = read_requests(args.file)
    except (OSError, ValueError) as e:
        log.error("%s", e)
        return 1
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        requests = [r for r in requests if r["id"] in wanted]
    else:
        requests = [r for r in requests if r["run"] in ("yes", "y", "1", "true", "x")]
    if not requests:
        log.error("No rows to run in %s", args.file.name)
        return 1

    results: list[dict] = []
    log.info("Batch: %s requests%s%s", len(requests), " (plan only)" if args.plan_only else "",
             " + review" if args.review else "")
    for n, req in enumerate(requests, 1):
        log.info("=== [%s/%s] id %s: %s %s", n, len(requests), req["id"], req["type"],
                 req.get("topic") or f"recording {req.get('recording')}")
        try:
            results.append(run_one(req, args.plan_only, args.review))
        except llm.BudgetExceeded as e:
            log.error("Batch stopped by the spending lock: %s", e)
            results.append({**{k: req.get(k, "") for k in ("id", "type", "topic", "recording")},
                            "status": "STOPPED", "problems": str(e)[:300]})
            break

    out = config.OUTPUT_DIR / f"batch_{datetime.now():%Y%m%d-%H%M%S}.csv"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    page = report.build()

    print(f"\n{'id':>4}  {'status':8}  {'cost':>7}  {'time':>6}  request / first problem")
    for r in results:
        what = r.get("topic") or f"recording {r.get('recording')}"
        print(f"{r['id']:>4}  {r['status']:8}  ${float(r.get('cost_usd_est') or 0):>6.3f}  {r.get('seconds', '-'):>5}s  "
              f"{what[:40]}  {('| ' + r['problems'][:70]) if r.get('problems') else ''}")
    total = sum(float(r.get("cost_usd_est") or 0) for r in results)
    print(f"\nEstimated cost of this batch: ${total:.3f}. Today so far: ${llm.spent_today():.2f}")
    print(f"Summary: {out}\nReport:  {page}  (open it in the browser)")
    return 0 if all(r["status"] in ("OK", "CHECK", "PLAN") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
