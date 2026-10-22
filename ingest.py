"""
ingest.py - turns Teams transcripts (.vtt or .docx) into ~1-minute segments.

Why: a Teams VTT has a new cue every 2-5 seconds, often mid-sentence, so cues are too
small to search and too many to send to a model. We merge them into paragraphs of about
a minute, cut at a sentence end, and keep who said what and exactly when. Every later
step (slides, search, lessons, video clips) works on these segments.

Usage:
  uv run ingest.py --file "data/input/Weekly sync.vtt"   # test one file, saves nothing
  uv run ingest.py                                        # all manifest rows with include=yes
  uv run ingest.py --force                                # redo already processed videos
"""
from __future__ import annotations

import argparse
import html
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import config
import store

log = logging.getLogger("ingest")

TARGET_SEC = 60      # close a segment after ~1 minute, at the next sentence end
MAX_SEC = 120        # never longer than 2 minutes, even without a sentence end
MAX_GAP_SEC = 30     # silence longer than this always starts a new segment
MIN_TAIL_WORDS = 25  # a tiny last segment is merged into the one before

_TS = r"\d{1,2}(?::\d{1,2}){1,2}(?:[.,]\d{1,3})?"          # 00:01:02.345, 1:02.3, 0:0:5,4
TIMING = re.compile(rf"^\s*({_TS})\s*-->\s*({_TS})")          # VTT timing line
VOICE = re.compile(r"<v(?:\.[^\s>]+)?\s+([^>]+)>")            # Teams speaker tag <v Jane Doe>
TAG = re.compile(r"</?[^>]+>")
SENTENCE_END = re.compile(r"[.!?…][\"')\]]*\s*$")
DOCX_HEADER = re.compile(rf"^(?P<speaker>.+?)\s+(?P<ts>{_TS})\s*$")  # "Jane Doe   0:03"
SYSTEM_LINE = re.compile(r"\b(started|stopped) transcription\b", re.I)
NAME_PARTICLES = {"van", "von", "der", "den", "de", "da", "di", "du", "la", "le", "y", "zu"}


@dataclass
class Cue:
    start: float
    end: float
    speaker: str
    text: str


def to_seconds(ts: str) -> float:
    """'01:02:03.450', '2:03.4' or '0:0:5,430' -> seconds."""
    seconds = 0.0
    for part in ts.replace(",", ".").split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def clean(text: str) -> str:
    """Remove tags, decode &amp; etc., collapse whitespace."""
    return " ".join(html.unescape(TAG.sub("", text)).split())


def _looks_like_name(line: str) -> bool:
    """'Doe, Jane (Roche)' -> True; 'We meet at' -> False. Used for .docx speaker lines."""
    words = line.split()
    if not 1 <= len(words) <= 6 or line.rstrip()[-1:] in ".?!;:,":
        return False
    return all(w[0].isupper() or w.lower() in NAME_PARTICLES
               for w in words if w[0].isalpha())


def _finish(cues: list[Cue]) -> list[Cue]:
    """Sort, fill missing end times, drop exact repeats."""
    cues.sort(key=lambda c: c.start)
    for i, c in enumerate(cues):
        if c.end <= c.start:   # .docx has only start times: end where the next cue starts
            nxt = cues[i + 1].start if i + 1 < len(cues) else c.start + len(c.text.split()) / 2.5
            c.end = max(nxt, c.start + 1)
    out: list[Cue] = []
    for c in cues:
        if out and c.text == out[-1].text and c.speaker == out[-1].speaker:
            out[-1].end = max(out[-1].end, c.end)
            continue
        out.append(c)
    return out


def parse_vtt(path: Path) -> list[Cue]:
    """Teams .vtt: blocks of [cue id] / timing line / '<v Speaker>text</v>'."""
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line for line in block.strip().split("\n") if line.strip()]
        for i, line in enumerate(lines):
            m = TIMING.match(line)
            if not m:
                continue
            raw = " ".join(lines[i + 1:])
            voice = VOICE.search(raw)
            body = clean(raw)
            if body:
                cues.append(Cue(to_seconds(m.group(1)), to_seconds(m.group(2)),
                                clean(voice.group(1)) if voice else "", body))
            break
    return _finish(cues)


def parse_docx(path: Path) -> list[Cue]:
    """
    Teams .docx transcripts come in two layouts; both are handled:
      new:  'Jane Doe   0:03'  then the text paragraph(s)
      old:  '0:0:3.4 --> 0:0:5.1'  then 'Jane Doe'  then the text
    Title, date and duration lines before the first cue are ignored.
    """
    from docx import Document   # imported here so .vtt-only setups don't need python-docx

    lines = [" ".join(p.text.split()) for p in Document(str(path)).paragraphs]
    lines = [line for line in lines if line and not SYSTEM_LINE.search(line)]
    cues: list[Cue] = []
    cur: Cue | None = None
    expect_speaker = False

    for line in lines:
        timing = TIMING.match(line)
        header = DOCX_HEADER.match(line)
        if timing:
            if cur and cur.text:
                cues.append(cur)
            cur = Cue(to_seconds(timing.group(1)), to_seconds(timing.group(2)), "", "")
            expect_speaker = True
        elif header and _looks_like_name(header.group("speaker")):
            if cur and cur.text:
                cues.append(cur)
            cur = Cue(to_seconds(header.group("ts")), 0.0, clean(header.group("speaker")), "")
            expect_speaker = False
        elif cur is not None:
            if expect_speaker and not cur.text and _looks_like_name(line):
                cur.speaker = clean(line)
            else:
                cur.text = f"{cur.text} {clean(line)}".strip()
            expect_speaker = False
    if cur and cur.text:
        cues.append(cur)
    return _finish(cues)


def parse_transcript(path: Path) -> list[Cue]:
    """Read a .vtt or .docx transcript into time-stamped cues."""
    suffix = path.suffix.lower()
    if suffix == ".vtt":
        return parse_vtt(path)
    if suffix == ".docx":
        return parse_docx(path)
    raise ValueError(f"Unsupported transcript type: {path.name} (use .vtt or .docx)")


def build_segments(cues: list[Cue], video_id: str) -> list[dict]:
    """Group cues into ~1-minute segments (rules: TARGET_SEC, MAX_SEC, MAX_GAP_SEC)."""
    groups: list[list[Cue]] = []
    cur: list[Cue] = []
    for cue in cues:
        if cur and (cue.end - cur[0].start > MAX_SEC or cue.start - cur[-1].end > MAX_GAP_SEC):
            groups.append(cur)
            cur = []
        cur.append(cue)
        if cue.end - cur[0].start >= TARGET_SEC and SENTENCE_END.search(cue.text):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    if len(groups) > 1 and sum(len(c.text.split()) for c in groups[-1]) < MIN_TAIL_WORDS:
        groups[-2].extend(groups.pop())
    return [_to_segment(g, video_id, i) for i, g in enumerate(groups)]


def _to_segment(group: list[Cue], video_id: str, idx: int) -> dict:
    """One segment: consecutive lines of the same speaker are joined into one turn."""
    turns: list[list[str]] = []
    for c in group:
        if turns and turns[-1][0] == c.speaker:
            turns[-1][1] += " " + c.text
        else:
            turns.append([c.speaker, c.text])
    return {
        "id": f"{video_id}#{idx:04d}",
        "video_id": video_id,
        "idx": idx,
        "start": round(group[0].start, 2),
        "end": round(max(c.end for c in group), 2),
        "speakers": list(dict.fromkeys(s for s, _ in turns if s)),
        "text": "\n".join(f"{s}: {t}" if s else t for s, t in turns),
        "words": sum(len(c.text.split()) for c in group),
        "slide_ids": [],   # filled in by slides.py
    }


def describe(cues: list[Cue], segments: list[dict]) -> str:
    """Quick health report for one transcript."""
    if not segments:
        return "no segments"
    durs = [s["end"] - s["start"] for s in segments]
    no_speaker = sum(not c.speaker for c in cues) / max(len(cues), 1)
    lines = [f"{len(cues)} cues -> {len(segments)} segments, "
             f"avg {sum(durs) / len(durs):.0f}s / {sum(s['words'] for s in segments) / len(segments):.0f} words, "
             f"transcript ends at {store.fmt_ts(segments[-1]['end'])}",
             f"speakers: {', '.join(sorted({sp for s in segments for sp in s['speakers']})) or 'none found'}"]
    if no_speaker > 0.5:
        lines.append("WARNING: most cues have no speaker - check the file format")
    return "\n".join(lines)


def ingest_row(row: dict, force: bool) -> str:
    """Turn the transcript of one manifest row into ~1-minute segments plus meta.json.
    Returns 'done', 'skipped' (already done and not forced) or 'no transcript'."""
    vid = row["video_id"]
    path = store.transcript_path(row)   # downloaded transcript, else one made by transcribe.py
    if path is None:
        log.warning("%s: no transcript yet (run in mode 'all' to transcribe it), skipped",
                    row.get("video_file"))
        return "no transcript"
    if store.has_segments(vid) and not force:
        return "skipped"
    cues = parse_transcript(path)
    if not cues:
        raise ValueError(f"No cues found in {path.name}. Is it a Teams transcript?")
    segments = build_segments(cues, vid)
    if store.has_slides(vid):   # re-ingested: keep the links to slides found earlier
        from slides import link
        slides = store.load_slides(vid)
        link(slides, segments)
        store.save_slides(vid, slides)
    store.save_segments(vid, segments)
    store.save_meta(vid, {
        "video_id": vid, "title": row.get("title") or vid,
        "catalog_id": row.get("catalog_id", ""), "description": row.get("description", ""),
        "media_type": row.get("media_type", "video"), "crop": row.get("crop", ""),
        "video_file": row.get("video_file", ""), "transcript_file": str(path),
        "transcript_end_s": segments[-1]["end"], "n_segments": len(segments),
        "speakers": sorted({sp for s in segments for sp in s["speakers"]}),
    })
    log.info("%s: %s", vid, describe(cues, segments).replace("\n", " | "))
    return "done"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, help="test a single transcript and print the result")
    ap.add_argument("--force", action="store_true", help="re-process videos already done")
    args = ap.parse_args()
    config.setup_logging()

    if args.file:
        cues = parse_transcript(args.file)
        segments = build_segments(cues, "test")
        print(describe(cues, segments))
        for s in segments[:3]:
            print(f"\n--- {s['id']}  {store.fmt_ts(s['start'])}-{store.fmt_ts(s['end'])}  ({s['words']} words)")
            print(s["text"][:600])
        return 0

    rows = store.included_rows()
    if not rows:
        log.error("No manifest rows with include=yes. Run inventory.py and edit data/manifest.csv")
        return 1
    counts: dict[str, int] = {}
    for row in rows:
        try:
            result = ingest_row(row, args.force)
        except Exception as e:  # one bad file must not stop the batch
            log.error("%s: FAILED - %s", row.get("video_file"), e)
            result = "failed"
        counts[result] = counts.get(result, 0) + 1
    log.info("Finished: %s", ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
