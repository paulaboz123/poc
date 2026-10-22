"""
slides.py - finds the slides shown in a Teams recording, reads their text and links each
slide to the transcript segments spoken while it was on screen.

How it works, per video:
 1. ffmpeg samples one small greyscale frame every 2 s (~15 MB of temp files per hour).
 2. Each frame is compared with the previous one on a 16x9 grid. Teams puts small moving
    webcam tiles on top of a shared screen; those change only a few grid cells. A new slide
    is declared only when a large share of the cells changes at once.
 3. Each stable stretch ("run") is one slide candidate. Runs shorter than 6 s (transitions,
    flicker, scrolling) are dropped. We keep the LAST frame of each run, so a slide that
    builds up bullet by bullet is captured complete.
 4. ffmpeg grabs that moment again in full resolution and Tesseract reads the text.
    Candidates with almost no text (gallery view, people talking) get is_slide = false.
 5. Slides and segments are linked by time overlap (segment.slide_ids, slide.segment_ids).

Teams meeting recordings often show the participants' tiles next to the slide. Set a crop
(SLIDE_CROP in .env, or the 'crop' column of a manifest row) so only the slide area is used.
Check it with --preview (red box = slide area). If a presenter's camera picture sits ON the
slide, set SLIDE_MASK too (blue box): that part is painted over in the video lesson.

Usage:
  uv run slides.py --video <video_id> --preview   # check the crop on one frame
  uv run slides.py --video <video_id>    # one video first; look at data/processed/<id>/slides/
  uv run slides.py                       # every ingested video that has a video file
  uv run slides.py --force               # redo
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

import config
import media
import store

log = logging.getLogger("slides")

SAMPLE_EVERY_S = 2     # one frame every 2 s
FRAME_W, FRAME_H = 320, 180
GRID_COLS, GRID_ROWS = 16, 9
PIXEL_DIFF = 40        # grey levels (0-255) a pixel must change by to count as changed
CELL_CHANGED = 0.03    # a cell counts as changed if >3% of its pixels changed
NEW_SLIDE_FRAC = 0.15  # new slide if >15% of cells changed. Raise if you get duplicates,
                       # lower if slides are missing (e.g. same template, little text change)
MIN_SLIDE_S = 6        # shorter runs are transitions or scrolling
MIN_TEXT_CHARS = 25    # fewer letters/digits than this = not a slide (webcams, gallery)


def _load(path: Path) -> np.ndarray:
    """Frame as a grey-level array (int16, so differences between frames can be negative)."""
    with Image.open(path) as im:          # 'with' closes the file (Windows can't delete open files)
        return np.asarray(im.convert("L"), dtype=np.int16)


def change_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Share of grid cells (0..1) that changed between two frames."""
    changed = np.abs(a - b) > PIXEL_DIFF
    h, w = changed.shape
    h, w = h - h % GRID_ROWS, w - w % GRID_COLS
    cells = changed[:h, :w].reshape(GRID_ROWS, h // GRID_ROWS, GRID_COLS, w // GRID_COLS)
    return float((cells.mean(axis=(1, 3)) > CELL_CHANGED).mean())


def find_runs(frames: list[Path]) -> list[tuple[int, int]]:
    """Split the frame sequence into stable stretches: [(first_index, last_index), ...]."""
    runs, start, prev = [], 0, _load(frames[0])
    for i in range(1, len(frames)):
        cur = _load(frames[i])
        if change_fraction(prev, cur) > NEW_SLIDE_FRAC:
            runs.append((start, i - 1))
            start = i
        prev = cur
    runs.append((start, len(frames) - 1))
    return runs


def link(slides: list[dict], segments: list[dict]) -> None:
    """Connect slides and segments that overlap in time (in place)."""
    for seg in segments:
        seg["slide_ids"] = []
    for sl in slides:
        sl["segment_ids"] = []
        if not sl["is_slide"]:
            continue
        for seg in segments:
            if seg["start"] < sl["end"] and seg["end"] > sl["start"]:
                sl["segment_ids"].append(seg["id"])
                seg["slide_ids"].append(sl["id"])


def process_video(video_id: str, force: bool) -> str:
    """Find, save and read the slides of one recording and link them to its segments.
    Returns 'done', 'skipped' (already done and not forced) or 'no video'."""
    meta = store.load_meta(video_id)
    if not meta.get("video_file") or meta.get("media_type") == "audio":
        return "no video"
    if store.has_slides(video_id) and not force:
        return "skipped"
    video = config.INPUT_DIR / meta["video_file"]
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    t0 = time.perf_counter()
    crop = media.parse_crop(meta.get("crop") or config.SLIDE_CROP)
    duration = media.duration_s(video)
    out_dir = store.slides_dir(video_id)
    shutil.rmtree(out_dir, ignore_errors=True)
    store.video_dir(video_id).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=store.video_dir(video_id), ignore_cleanup_errors=True) as tmp:
        frames = media.sample_frames(video, Path(tmp), SAMPLE_EVERY_S, FRAME_W, FRAME_H, crop)
        if not frames:
            raise media.MediaError("ffmpeg produced no frames - is the file a video?")
        runs = find_runs(frames)

    kept = [r for r in runs if (r[1] - r[0] + 1) * SAMPLE_EVERY_S >= MIN_SLIDE_S]
    slides = []
    for i, (first, last) in enumerate(kept):
        start = first * SAMPLE_EVERY_S
        end = kept[i + 1][0] * SAMPLE_EVERY_S if i + 1 < len(kept) else duration
        at = min(last * SAMPLE_EVERY_S, max(duration - 0.5, 0))
        image = media.grab_frame(video, at, out_dir / f"slide_{i:03d}.jpg", crop)
        text = media.ocr(image)
        slides.append({
            "id": f"{video_id}@{i:03d}", "video_id": video_id,
            "start": float(start), "end": round(float(end), 2), "frame_at": float(at),
            "file": image.relative_to(store.video_dir(video_id)).as_posix(),
            "text": text, "is_slide": sum(ch.isalnum() for ch in text) >= MIN_TEXT_CHARS,
            "segment_ids": [],
        })

    segments = store.load_segments(video_id)
    link(slides, segments)
    store.save_slides(video_id, slides)
    store.save_segments(video_id, segments)
    n_real = sum(s["is_slide"] for s in slides)
    log.info("%s: %s frames -> %s candidates -> %s slides with text (%.0fs)",
             video_id, len(frames), len(slides), n_real, time.perf_counter() - t0)
    if n_real == 0:
        log.warning("%s: no slides found - camera-only meeting, or NEW_SLIDE_FRAC too high", video_id)
    return "done"


def preview_crop(video_id: str, at_s: float = 60) -> Path:
    """Save one full frame with the crop area drawn in red, to check the crop setting."""
    from PIL import ImageDraw
    meta = store.load_meta(video_id)
    video = config.INPUT_DIR / meta["video_file"]
    out = store.video_dir(video_id) / "crop_preview.jpg"
    media.grab_frame(video, min(at_s, max(media.duration_s(video) - 1, 0)), out)
    crop = media.parse_crop(meta.get("crop") or config.SLIDE_CROP) or (0, 0, 1, 1)
    with Image.open(out) as im:
        im = im.convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)
        draw.rectangle([crop[0] * w, crop[1] * h, crop[2] * w - 1, crop[3] * h - 1],
                       outline=(255, 0, 0), width=6)
        if config.SLIDE_MASK:   # blue: area painted over in the video lesson (inside the crop)
            m = media.parse_crop(config.SLIDE_MASK)
            cw, ch = (crop[2] - crop[0]) * w, (crop[3] - crop[1]) * h
            x0, y0 = crop[0] * w, crop[1] * h
            draw.rectangle([x0 + m[0] * cw, y0 + m[1] * ch, x0 + m[2] * cw - 1, y0 + m[3] * ch - 1],
                           outline=(0, 90, 255), width=6)
        im.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="video_id from the manifest (default: all)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preview", action="store_true",
                    help="with --video: save crop_preview.jpg showing the crop area, then stop")
    args = ap.parse_args()
    config.setup_logging()

    if args.preview:
        if not args.video:
            log.error("--preview needs --video <video_id>")
            return 1
        log.info("Saved %s - red box: only the slide; blue box (SLIDE_MASK): area hidden in the video lesson",
                 preview_crop(args.video))
        return 0

    ids = [args.video] if args.video else store.processed_video_ids()
    if not ids:
        log.error("Nothing to do. Run ingest.py first.")
        return 1
    counts: dict[str, int] = {}
    for vid in ids:
        try:
            result = process_video(vid, args.force)
        except Exception as e:
            log.error("%s: FAILED - %s", vid, e)
            result = "failed"
        counts[result] = counts.get(result, 0) + 1
    log.info("Finished: %s", ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
