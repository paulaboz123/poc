"""
store.py - all reading and writing of the project's data lives here.

No other script opens data files directly. If the platform later wants its own storage
or vector store, this is the only file that changes.

Layout on disk:
  data/manifest.csv                       which video goes with which transcript
  data/processed/<video_id>/meta.json     title, files, duration, speakers
  data/processed/<video_id>/segments.json ~1-minute transcript segments
  data/processed/<video_id>/slides.json   detected slides + OCR text
  data/processed/<video_id>/slides/*.jpg  slide images
  data/index/                             embeddings for search
  data/output/<lesson_id>/                generated lessons
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import config

MANIFEST_FIELDS = ["video_id", "include", "catalog_id", "title", "description", "media_type", "video_file",
                   "transcript_file", "pair_method", "video_min", "transcript_min", "crop", "check"]


# --- Small helpers --------------------------------------------------------------
def slugify(text: str, max_len: int = 50) -> str:
    """'Budget Review – Q3 (final)' -> 'budget-review-q3-final'."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "item"


def stable_id(name: str, label: str = "") -> str:
    """Readable, unique, always the same for the same file: 'budget-review-2024-10-15-3f2a'."""
    return f"{slugify(label or Path(name).stem, 40)}-{hashlib.sha1(name.encode()).hexdigest()[:4]}"


def fmt_ts(seconds: float) -> str:
    """75 -> '01:15', 3725 -> '1:02:05'."""
    s = int(seconds)
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def write_json(path: Path, obj: Any) -> None:
    """Atomic write: write a temp file, then rename. A crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        _replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _replace(tmp: str | Path, target: Path) -> None:
    """os.replace with a readable error: on Windows it fails while Excel has the file open."""
    try:
        os.replace(tmp, target)
    except PermissionError as e:
        Path(tmp).unlink(missing_ok=True)
        raise PermissionError(f"Can't write {target.name}: close it in Excel/VS Code and run again") from e


def read_json(path: Path, default: Any = None) -> Any:
    """Parsed JSON file, or `default` if the file does not exist."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# --- Manifest --------------------------------------------------------------------
def read_manifest() -> list[dict]:
    """Read manifest.csv. Survives Excel re-saving it (semicolons, non-UTF-8 encoding)."""
    if not config.MANIFEST.exists():
        return []
    raw = config.MANIFEST.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")   # Excel "CSV (comma delimited)" on Windows
    first_line = text.splitlines()[0] if text else ""
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    rows = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items() if k} for r in rows]


def write_manifest(rows: list[dict]) -> None:
    """Write manifest.csv as UTF-8 with BOM, so Excel shows accents and umlauts correctly."""
    config.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    extra = [k for r in rows for k in r if k not in MANIFEST_FIELDS]
    fields = MANIFEST_FIELDS + list(dict.fromkeys(extra))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    tmp = config.MANIFEST.with_suffix(".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8-sig", newline="")
    _replace(tmp, config.MANIFEST)


def included_rows() -> list[dict]:
    """Manifest rows marked include=yes (your development set)."""
    yes = {"yes", "y", "1", "true", "x"}
    return [r for r in read_manifest() if r.get("include", "").lower() in yes]


# --- Per-video data -----------------------------------------------------------------
def video_dir(video_id: str) -> Path:
    """Folder with everything derived from one recording: data/processed/<video_id>."""
    return config.PROCESSED_DIR / video_id


def generated_transcript(video_id: str) -> Path:
    """Where transcribe.py writes a transcript it made (data/input is never modified)."""
    return video_dir(video_id) / "transcript.vtt"


def transcript_path(row: dict) -> Path | None:
    """The transcript to use for a manifest row: the downloaded one, else a generated one."""
    if row.get("transcript_file"):
        path = config.INPUT_DIR / row["transcript_file"]
        if path.exists():
            return path
    generated = generated_transcript(row["video_id"])
    return generated if generated.exists() else None


def slides_dir(video_id: str) -> Path:
    """Folder with the slide images of one recording."""
    return video_dir(video_id) / "slides"


def save_meta(video_id: str, meta: dict) -> None:
    """Save meta.json (title, files, duration, speakers) of one recording."""
    write_json(video_dir(video_id) / "meta.json", meta)


def load_meta(video_id: str) -> dict:
    """Load meta.json of one recording; raises if it was never ingested."""
    meta = read_json(video_dir(video_id) / "meta.json")
    if meta is None:
        raise FileNotFoundError(f"No meta.json for {video_id}. Run ingest.py first.")
    return meta


def save_segments(video_id: str, segments: list[dict]) -> None:
    """Save the transcript segments of one recording."""
    write_json(video_dir(video_id) / "segments.json", segments)


def load_segments(video_id: str) -> list[dict]:
    """Transcript segments of one recording ([] if not ingested yet)."""
    return read_json(video_dir(video_id) / "segments.json", [])


def has_segments(video_id: str) -> bool:
    """True if the recording has been ingested."""
    return (video_dir(video_id) / "segments.json").exists()


def save_slides(video_id: str, slides: list[dict]) -> None:
    """Save the detected slides (with OCR text) of one recording."""
    write_json(video_dir(video_id) / "slides.json", slides)


def load_slides(video_id: str) -> list[dict]:
    """Detected slides of one recording ([] if none yet)."""
    return read_json(video_dir(video_id) / "slides.json", [])


def has_slides(video_id: str) -> bool:
    """True if slide detection has run for the recording."""
    return (video_dir(video_id) / "slides.json").exists()


def processed_video_ids() -> list[str]:
    """Videos that have been ingested (have segments), restricted to include=yes if a
    manifest exists, so unticking a video in the manifest removes it from search."""
    if not config.PROCESSED_DIR.exists():
        return []
    ids = sorted(p.parent.name for p in config.PROCESSED_DIR.glob("*/segments.json"))
    included = {r["video_id"] for r in included_rows()}
    return [i for i in ids if i in included] if included else ids


def all_segments() -> dict[str, dict]:
    """Every segment of every processed video, by id, with title and slide text attached."""
    out: dict[str, dict] = {}
    for vid in processed_video_ids():
        meta = load_meta(vid)
        slides = {s["id"]: s for s in load_slides(vid)}
        for seg in load_segments(vid):
            seg = dict(seg)
            seg["title"] = meta.get("title", vid)
            seg["description"] = meta.get("description", "")   # from description.xlsx, may be empty
            # slide text = what OCR read, plus the vision model's description when there is one
            seg["slide_text"] = "\n".join(
                "\n".join(x for x in (slides[i]["text"], slides[i].get("visual", "")) if x)
                for i in seg.get("slide_ids", []) if i in slides and slides[i].get("is_slide"))
            out[seg["id"]] = seg
    return out


def available_recordings() -> list[dict]:
    """Included recordings that have been processed (ready for lessons and videos)."""
    return [r for r in included_rows() if has_segments(r["video_id"])]


def resolve_recordings(ref: str) -> list[str]:
    """Recordings for a user reference: Excel id ('13'), video_id, or part of the title or of
    the file name (case does not matter). Several refs separated by commas: '13,joining'."""
    rows = available_recordings()
    if not rows:
        raise ValueError("No processed recordings yet: set include=yes in data/manifest.csv and run pipeline.py.")
    found: list[str] = []
    for part in (x.strip() for x in ref.split(",") if x.strip()):
        low = part.lower()
        match = ([r for r in rows if r.get("catalog_id") == part] or
                 [r for r in rows if r["video_id"] == part] or
                 [r for r in rows if low in r.get("title", "").lower()] or
                 [r for r in rows if low in (r.get("video_file") or r.get("transcript_file") or "").lower()])
        if not match:
            listing = "\n".join(f"  {r.get('catalog_id') or '-':>4}  {r['title']}" for r in rows)
            raise ValueError(f"No processed recording matches '{part}'. Use the id, or a word from the "
                             f"title or file name. Available recordings (id, title):\n{listing}")
        found += [r["video_id"] for r in match if r["video_id"] not in found]
    return found


# --- Search index ----------------------------------------------------------------
def save_index(vectors, ids: list[str], info: dict) -> None:
    """Save the search index: vectors, the segment id of each row, and build info."""
    import numpy as np
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_DIR / "vectors.npy", vectors.astype("float32"))
    write_json(config.INDEX_DIR / "ids.json", ids)
    write_json(config.INDEX_DIR / "info.json", info)


def load_index():
    """Returns (vectors, ids, info) or raises if the index was never built."""
    import numpy as np
    path = config.INDEX_DIR / "vectors.npy"
    if not path.exists():
        raise FileNotFoundError("No search index yet. Run: uv run index.py build")
    return (np.load(path), read_json(config.INDEX_DIR / "ids.json"),
            read_json(config.INDEX_DIR / "info.json", {}))


# --- Skills and lessons ----------------------------------------------------------
def load_skill(name: str) -> str:
    """Body of skills/<name>/<latest version>/SKILL.md, without the YAML front matter."""
    base = config.SKILLS_DIR / name
    versions = sorted((p for p in base.iterdir() if (p / "SKILL.md").exists()),
                      key=lambda p: [int(x) if x.isdigit() else x for x in p.name.split(".")]
                      ) if base.exists() else []
    path = versions[-1] / "SKILL.md" if versions else base / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"Skill not found: {path}")
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        text = parts[2] if len(parts) == 3 else text
    return text.strip()


def save_lesson(lesson_id: str, lesson: dict, markdown: str) -> Path:
    """Save a text lesson as lesson.json (all data) and lesson.md (readable). Returns its folder."""
    out = config.OUTPUT_DIR / lesson_id
    write_json(out / "lesson.json", lesson)
    (out / "lesson.md").write_text(markdown, encoding="utf-8")
    return out
