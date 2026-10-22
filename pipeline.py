"""
pipeline.py - runs the whole preparation in the right order, for the mode you choose.

Modes (pick with --mode, or PIPELINE_MODE in .env; --mode wins):
  easy   videos that already have a transcript                     <- start here
  audio  + audio files and standalone transcripts with a transcript
  all    + every recording without a transcript (transcribed locally with Whisper)
Add --vision to also describe what each slide shows (diagrams, processes). Recommended
with the gateway (cheap, fast); slow with a local model.

Order of steps, each skipping what is already done:
  1. inventory        find recordings, pair them with transcripts  -> data/manifest.csv
  2. transcribe       only in mode "all": make missing transcripts
  3. ingest           transcripts -> ~1-minute segments
  4. slides           videos only: detect slides, read their text
  5. describe slides  only with --vision
  6. index            rebuild the search index

Only manifest rows with include=yes are processed, in every mode. A row the current mode
does not cover is listed at the end, so nothing is skipped silently.

Usage:
  uv run pipeline.py                        # mode from .env (default: easy)
  uv run pipeline.py --mode audio
  uv run pipeline.py --mode all --vision
  uv run pipeline.py --force                # redo ingest + slides for the selected rows
"""
from __future__ import annotations

import argparse
import logging
import sys

import config
import index
import ingest
import inventory
import llm
import slides
import store

log = logging.getLogger("pipeline")


def in_mode(row: dict, mode: str) -> tuple[bool, str]:
    """Is this row handled in this mode? Returns (yes/no, reason if no)."""
    kind = row.get("media_type") or ("video" if row.get("video_file") else "none")
    has_transcript = store.transcript_path(row) is not None
    if mode == "all":
        return True, ""
    if not has_transcript:
        return False, "no transcript (needs mode 'all')"
    if kind == "video":
        return True, ""
    if mode == "audio":
        return True, ""
    label = "transcript without a recording" if kind == "none" else f"{kind} file"
    return False, f"{label} (needs mode 'audio')"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=config.MODES, default=config.PIPELINE_MODE)
    ap.add_argument("--vision", action="store_true", help="also describe slides with the vision model")
    ap.add_argument("--force", action="store_true", help="redo ingest and slides for the selected rows")
    args = ap.parse_args()
    config.setup_logging()
    if args.mode not in config.MODES:
        log.error("Unknown mode %r, use one of %s", args.mode, config.MODES)
        return 1
    log.info("=== Mode: %s%s", args.mode, " + vision" if args.vision else "")

    log.info("=== 1/6 Inventory")
    if inventory.main() != 0:
        return 1
    included = store.included_rows()
    if not included:
        log.error("No rows with include=yes in %s. Mark your dev set and run again.", config.MANIFEST)
        return 1
    selected, left_out = [], []
    for row in included:
        ok, reason = in_mode(row, args.mode)
        (selected if ok else left_out).append((row, reason))
    rows = [r for r, _ in selected]
    log.info("%s of %s included rows are covered by mode '%s'", len(rows), len(included), args.mode)

    failed = 0
    if args.mode == "all":
        log.info("=== 2/6 Transcribe missing transcripts")
        import transcribe   # imported only here: needs faster-whisper
        failed += transcribe.run(rows).get("failed", 0)
    else:
        log.info("=== 2/6 Transcribe: skipped in mode '%s'", args.mode)

    log.info("=== 3/6 Ingest transcripts")
    for row in rows:
        try:
            ingest.ingest_row(row, args.force)
        except Exception as e:
            log.error("%s: ingest FAILED - %s", row.get("video_file") or row.get("transcript_file"), e)
            failed += 1

    log.info("=== 4/6 Slides (videos only)")
    for row in rows:
        if row.get("media_type", "video") != "video" or not store.has_segments(row["video_id"]):
            continue
        try:
            slides.process_video(row["video_id"], args.force)
        except Exception as e:
            log.error("%s: slides FAILED - %s", row["video_id"], e)
            failed += 1

    if args.vision:
        log.info("=== 5/6 Describe slides with %s", config.VISION_MODEL)
        import describe_slides
        for row in rows:
            if store.has_slides(row["video_id"]):
                try:
                    describe_slides.describe_video(row["video_id"], args.force, None)
                except llm.BudgetExceeded as e:   # no point trying the other recordings
                    log.error("Describe slides stopped: %s", e)
                    failed += 1
                    break
                except Exception as e:
                    log.error("%s: describe slides FAILED - %s", row["video_id"], e)
                    failed += 1
    else:
        log.info("=== 5/6 Describe slides: skipped (add --vision)")

    log.info("=== 6/6 Search index")
    try:
        if index.build() != 0:
            failed += 1
    except Exception as e:
        log.error("Index build FAILED - %s", e)
        failed += 1

    for row, reason in left_out:
        log.info("Not processed in this mode: %s - %s", row.get("title"), reason)
    log.info("=== Done: %s rows processed, %s failures", len(rows), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
