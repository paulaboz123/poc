"""
export_notebooklm.py - writes the prepared recordings as text files to upload to NotebookLM as
sources. NotebookLM does not accept MP4 files, so this is the "NotebookLM route": same input,
another tool for the output (audio overviews, quizzes, mind maps) next to our own pipeline.

One Markdown file per included recording: title, description, then the transcript minute by
minute with timestamps, and the text (and description, if made) of each slide when it appears.
Speaker names are left out unless you add --keep-speakers.

Before uploading: NotebookLM is a Google service outside the company platform. Upload only
what you are allowed to (ask Clara if unsure).

Usage:
  uv run export_notebooklm.py                 # one file per recording
  uv run export_notebooklm.py --combined      # plus one file with everything (one source)
Output: data/output/notebooklm/
"""
from __future__ import annotations

import argparse
import logging
import re
import sys

import config
import store

log = logging.getLogger("export_notebooklm")


def _strip_speakers(text: str, speakers: list[str]) -> str:
    """Remove 'Name: ' at the start of each line for the known speakers."""
    if not speakers:
        return text
    pattern = re.compile(r"^(?:" + "|".join(re.escape(s) for s in speakers) + r"):\s*", re.M)
    return pattern.sub("", text)


def recording_markdown(video_id: str, keep_speakers: bool = False) -> str:
    """One recording as a Markdown document: header, then segments with timestamps and slides."""
    meta = store.load_meta(video_id)
    slides = {s["id"]: s for s in store.load_slides(video_id) if s.get("is_slide")}
    lines = [f"# {meta.get('title', video_id)}", ""]
    if meta.get("description"):
        lines += [f"*{meta['description']}*", ""]
    shown: set[str] = set()
    for seg in store.load_segments(video_id):
        lines.append(f"## {store.fmt_ts(seg['start'])} - {store.fmt_ts(seg['end'])}")
        for sid in seg.get("slide_ids", []):
            if sid in slides and sid not in shown:
                shown.add(sid)
                slide_text = " / ".join(x.strip() for x in slides[sid]["text"].splitlines() if x.strip())
                lines.append(f"> **Slide on screen:** {slide_text}")
                if slides[sid].get("visual"):
                    lines.append(f"> **What the slide shows:** {' '.join(slides[sid]['visual'].split())}")
        text = seg["text"] if keep_speakers else _strip_speakers(seg["text"], meta.get("speakers", []))
        lines += ["", " ".join(text.split()), ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combined", action="store_true", help="also write one file with all recordings")
    ap.add_argument("--keep-speakers", action="store_true", help="keep speaker names")
    args = ap.parse_args()
    config.setup_logging()
    ids = store.processed_video_ids()
    if not ids:
        log.error("No processed recordings. Run pipeline.py first.")
        return 1
    out_dir = config.OUTPUT_DIR / "notebooklm"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    for vid in ids:
        text = recording_markdown(vid, args.keep_speakers)
        path = out_dir / f"{vid}.md"
        path.write_text(text, encoding="utf-8")
        docs.append(text)
        log.info("Wrote %s (%s words)", path.name, len(text.split()))
    if args.combined:
        (out_dir / "all_recordings.md").write_text("\n\n---\n\n".join(docs), encoding="utf-8")
        log.info("Wrote all_recordings.md")
    print(f"\nUpload the .md files in {out_dir} to NotebookLM as sources (Add source -> Upload).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
