"""
evaluate.py - runs every request in eval/golden_set.csv and scores the results.

Why: "is the output good?" must be a number by the demo, not a feeling. Every time you
change a prompt, a model or retrieval, run this and compare against the previous run.

Automatic scores (computed here):
  length_ratio      words / target words (1.0 = perfect, ok within 0.85-1.15)
  coverage          share of the must_mention facts that appear in the lesson
  invalid_citations sources the model invented (should always be 0)
  uncited_sections  sections that cite nothing (possible hallucination)
  seconds, words    speed and size, for the cost outlook

Manual scores (you fill in 1-5 in the results file): faithful, level_fit, useful.
Use eval/rubric.md so you score the same way every time.

Usage:
  uv run evaluate.py                    # all requests
  uv run evaluate.py --limit 3          # first 3, for a quick check
  uv run evaluate.py --model <another model id>   # compare models
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import config
import lesson as lesson_mod
import store

log = logging.getLogger("evaluate")

GOLDEN = config.EVAL_DIR / "golden_set.csv"
RESULT_FIELDS = ["id", "topic", "minutes", "level", "status", "words", "length_ratio",
                 "coverage", "missing_facts", "invalid_citations", "uncited_sections",
                 "best_search_score", "seconds", "lesson_path",
                 "faithful_1to5", "level_fit_1to5", "useful_1to5", "comments"]


def coverage(text: str, must_mention: str) -> tuple[float | None, list[str]]:
    """Share of the ';'-separated facts found in the text (case-insensitive)."""
    facts = [f.strip() for f in must_mention.split(";") if f.strip()]
    if not facts:
        return None, []
    missing = [f for f in facts if f.lower() not in text.lower()]
    return round(1 - len(missing) / len(facts), 2), missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model")
    args = ap.parse_args()
    config.setup_logging()

    if not GOLDEN.exists():
        log.error("Missing %s. Create it first (see the plan, step 7).", GOLDEN)
        return 1
    with open(GOLDEN, encoding="utf-8-sig", newline="") as f:
        requests = [r for r in csv.DictReader(f) if r.get("topic", "").strip()]
    requests = requests[: args.limit] if args.limit else requests

    out_dir = config.EVAL_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{datetime.now():%Y%m%d-%H%M%S}_{store.slugify(args.model or config.CHAT_MODEL, 60)}.csv"

    results = []
    for n, req in enumerate(requests, 1):
        log.info("[%s/%s] %s", n, len(requests), req["topic"])
        row = {k: req.get(k, "") for k in ("id", "topic", "minutes", "level")}
        try:
            rec = lesson_mod.make_lesson(req["topic"], int(req.get("minutes") or 5),
                                         req.get("level") or "beginner", model=args.model)
            st = rec["stats"]
            text = lesson_mod.lesson_text(lesson_mod.Lesson(**rec["lesson"]))
            cov, missing = coverage(text, req.get("must_mention", ""))
            row.update(status="ok", words=st["words"], length_ratio=st["length_ratio"],
                       coverage="" if cov is None else cov, missing_facts="; ".join(missing),
                       invalid_citations=len(st["invalid_citations"]),
                       uncited_sections=len(st["sections_without_sources"]),
                       best_search_score=st["best_search_score"], seconds=st["seconds"],
                       lesson_path=rec["path"])
        except lesson_mod.NotEnoughMaterial as e:
            row.update(status="no material", comments=str(e)[:200])
        except Exception as e:
            log.error("Failed: %s", e)
            row.update(status="error", comments=str(e)[:200])
        results.append(row)
        _write(out_path, results)   # write after every request, so a crash loses nothing

    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        avg = lambda k: sum(float(r[k]) for r in ok if r[k] != "") / max(1, sum(r[k] != "" for r in ok))
        log.info("Done: %s/%s ok | length ratio %.2f | coverage %.2f | invalid citations %s | %.0fs avg",
                 len(ok), len(results), avg("length_ratio"), avg("coverage"),
                 sum(int(r["invalid_citations"]) for r in ok), avg("seconds"))
    log.info("Results: %s  -> fill in the *_1to5 columns", out_path)
    return 0


def _write(path: Path, rows: list[dict]) -> None:
    """Save the results table as CSV (opens in Excel); skipped while Excel has it open."""
    try:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    except PermissionError:
        log.warning("%s is open in Excel - results will be written after the next request", path.name)


if __name__ == "__main__":
    sys.exit(main())
