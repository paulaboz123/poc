"""
make_video.py - end to end: a request in, a short narrated video lesson out.

  request: a topic (searched in all recordings) or one recording, plus minutes and level
   1. material    the transcript excerpts and slides that cover it
   2. storyboard  the chat model plans the scenes: narration + what the screen shows
   3. voice       each scene's narration spoken locally (Kokoro)
   4. pictures    each scene drawn as an image: original slide, key points or steps
   5. video       ffmpeg puts pictures and voice together: lesson_video.mp4 + .srt subtitles,
                  and the narration alone as lesson_audio.mp3 (the audio lesson)

Usage:
  uv run make_video.py "joining data" --minutes 2
  uv run make_video.py --recording 13 --minutes 3 --level intermediate   # id from description.xlsx
  uv run make_video.py "joining data" --plan-only           # storyboard only, to check it first
  uv run make_video.py --rerender data/output/<folder>      # after editing storyboard.json by hand
  uv run make_video.py "joining data" --check               # plus the quality checks (quality.py)
  Many at once from a list: batch.py

Output in data/output/<time>-<topic>-video-<n>m/:
  lesson_video.mp4   the video (subtitles included as a track)
  lesson_video.srt   subtitles (also for editing or translation)
  lesson_audio.mp3   audio-only lesson
  script.md          the narration and on-screen text, scene by scene, with sources
  storyboard.json    the same as data: edit it and use --rerender
  scenes/, audio/    the image and the voice of each scene
  info.json          timings, word counts, estimated model cost
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

import config
import lesson
import llm
import media
import render
import store
import storyboard
import tts

log = logging.getLogger("make_video")


# --- 1 + 2: material and storyboard -------------------------------------------------------------
def plan(topic: str | None, recording: str | None, minutes: float, level: str,
         model: str | None) -> Path:
    """Find the material, ask for a storyboard, save it. Returns the output folder."""
    summary_of = ""
    if recording:
        video_ids = store.resolve_recordings(recording)
        segments = lesson.recording_segments(video_ids, int(round(minutes)))
        titles = list(dict.fromkeys(s["title"] for s in segments))
        summary_of = ", ".join(f'"{t}"' for t in titles)
        topic = topic or "; ".join(titles)
    else:
        segments, best = lesson.retrieve(topic, int(round(minutes)))
        log.info("Search: best match %.2f", best)
    if not segments:
        raise lesson.NotEnoughMaterial("No transcript segments found. Run pipeline.py first.")
    cost_before = llm.spent_this_run()
    sb = storyboard.make(topic, minutes, level, segments, model=model, summary_of=summary_of)
    sb["request"] = {"topic": topic, "recording": recording or "", "minutes": minutes, "level": level}
    sb["plan_cost_usd_est"] = round(llm.spent_this_run() - cost_before, 4)
    out_dir = config.OUTPUT_DIR / f"{sb['created']}-{store.slugify(topic, 40)}-video-{minutes:g}m"
    storyboard.save(sb, out_dir)
    log.info("Storyboard: %s scenes, %s words (target %s) -> %s", sb["stats"]["scenes"],
             sb["stats"]["words"], sb["stats"]["target_words"], out_dir / "script.md")
    return out_dir


# --- subtitles ---------------------------------------------------------------------------------
def _srt_time(t: float) -> str:
    """Seconds -> SRT timestamp, e.g. 62.345 -> '00:01:02,345'."""
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _chunks(text: str, max_words: int = 12) -> list[str]:
    """Split narration into subtitle-sized pieces at sentence ends, then commas, then length."""
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sentence.split()
        while len(words) > max_words:
            cut = next((i + 1 for i in range(max_words - 1, 3, -1) if words[i].endswith((",", ";", ":"))),
                       max_words)
            out.append(" ".join(words[:cut]))
            words = words[cut:]
        if words:
            out.append(" ".join(words))
    return out


def subtitles(scenes: list[dict], starts: list[float], durations: list[float]) -> str:
    """SRT with each scene's narration spread over its speaking time (by length of text)."""
    lines, n = [], 0
    for sc, start, dur in zip(scenes, starts, durations):
        pieces = _chunks(tts.clean_text(sc["narration"]))
        speak_from, speak_to = start + 0.15, start + max(dur - tts.PAUSE_AFTER_S, 0.5)
        total = sum(len(p) for p in pieces) or 1
        t = speak_from
        for p in pieces:
            end = t + (speak_to - speak_from) * len(p) / total
            n += 1
            lines += [str(n), f"{_srt_time(t)} --> {_srt_time(end)}", p, ""]
            t = end
    return "\n".join(lines)


# --- 3-5: voice, pictures, video ------------------------------------------------------------------
def produce(out_dir: Path) -> dict:
    """Voice + pictures + video for the storyboard in out_dir. No model calls."""
    sb = storyboard.load(out_dir)
    scenes = sb["scenes"]
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    wavs, durations = [], []
    for i, sc in enumerate(scenes, 1):
        wav = out_dir / "audio" / f"scene_{i:02d}.wav"
        durations.append(tts.speak_to_wav(sc["narration"], wav))
        wavs.append(wav)
        log.info("voice %s/%s: %.1fs", i, len(scenes), durations[-1])
    timings["voice_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    images = []
    for i, sc in enumerate(scenes, 1):
        slide = Path(sc["slide_image"]) if sc.get("slide_image") else None
        if sc["visual"] == "slide" and (slide is None or not slide.exists()):
            log.warning("scene %s: slide image missing (%s), showing key points", i, slide)
        images.append(render.render_scene(sc, out_dir / "scenes" / f"scene_{i:02d}.png",
                                          lesson_title=sb["title"], goal=sb.get("goal", ""),
                                          page=f"{i} / {len(scenes)}", slide_image=slide,
                                          caption=sc.get("caption", "")))
    timings["pictures_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    parts = [tts._read_wav(w) for w in wavs]
    rates = {r for _, r in parts}
    if len(rates) != 1:
        raise tts.TTSError(f"Scenes have different sample rates {rates}; use one TTS engine")
    rate = rates.pop()
    durations = [len(a) / rate for a, _ in parts]   # exact, from the samples
    full_wav = out_dir / "audio" / "narration.wav"
    tts._write_wav(full_wav, np.concatenate([a for a, _ in parts]), rate)
    starts = list(np.cumsum([0.0] + durations[:-1]))
    srt = out_dir / "lesson_video.srt"
    srt.write_text(subtitles(scenes, starts, durations), encoding="utf-8")
    video = media.stills_to_video(images, durations, full_wav, out_dir / "lesson_video.mp4", srt)
    audio = media.to_mp3(full_wav, out_dir / "lesson_audio.mp3")
    timings["video_s"] = round(time.perf_counter() - t0, 1)

    info = {"title": sb["title"], "video": str(video), "audio": str(audio),
            "duration_s": round(sum(durations), 1), "scenes": len(scenes),
            "words": sum(len(s["narration"].split()) for s in scenes),
            "tts": f"{config.TTS_ENGINE} {config.TTS_VOICE}", "timings": timings,
            "storyboard_stats": sb.get("stats", {})}
    return info


def make(topic: str | None = None, recording: str | None = None, minutes: float = 2,
         level: str = "beginner", model: str | None = None, plan_only: bool = False,
         rerender: Path | None = None) -> tuple[Path, dict]:
    """The whole flow for one request (or a re-render of an existing folder). Writes info.json
    and returns (output folder, info). Raises on errors; see main() for the readable handling."""
    t_start = time.perf_counter()
    out_dir = rerender if rerender else plan(topic, recording, minutes, level, model)
    sb = storyboard.load(out_dir)
    base = {"title": sb["title"], "request": sb.get("request", {}),
            "model": sb.get("model", ""), "model_cost_usd_est": sb.get("plan_cost_usd_est", 0.0)}
    if plan_only:
        info = {**base, "plan_only": True, "storyboard_stats": sb.get("stats", {})}
    else:
        info = {**base, **produce(out_dir)}
    info["total_s"] = round(time.perf_counter() - t_start, 1)
    store.write_json(out_dir / "info.json", info)
    return out_dir, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic", nargs="?", help="what the lesson is about (searched in all recordings)")
    ap.add_argument("--recording", help="make the video from one recording: Excel id (e.g. 13), "
                                        "video_id or part of the title; several: 13,14")
    ap.add_argument("--minutes", type=float, default=2, help="target length (default 2)")
    ap.add_argument("--level", choices=list(lesson.LEVELS), default="beginner")
    ap.add_argument("--model", help="override CHAT_MODEL from .env")
    ap.add_argument("--plan-only", action="store_true", help="stop after the storyboard")
    ap.add_argument("--rerender", type=Path, help="output folder: voice + video again from its storyboard.json")
    ap.add_argument("--check", action="store_true", help="run the quality checks afterwards (quality.py)")
    args = ap.parse_args()
    config.setup_logging()

    if not (args.topic or args.recording or args.rerender):
        ap.error("give a topic, --recording or --rerender")
    try:
        out_dir, info = make(args.topic, args.recording, args.minutes, args.level, args.model,
                             args.plan_only, args.rerender)
    except lesson.NotEnoughMaterial as e:
        log.error("%s", e)
        return 2
    except (RuntimeError, ValueError, OSError) as e:   # model, voice, ffmpeg, index or file problems
        log.error("%s", e)
        return 1

    if args.plan_only:
        print(f"\nStoryboard ready. Read {out_dir / 'script.md'}")
        print(f"Then make the video: uv run make_video.py --rerender \"{out_dir}\"")
        print(f"Estimated model cost: ${info['model_cost_usd_est']:.3f}")
        return 0
    m, s = divmod(int(info["duration_s"]), 60)
    print(f"\nVideo:  {info['video']}  ({m}:{s:02d}, {info['scenes']} scenes)")
    print(f"Audio:  {info['audio']}")
    print(f"Script: {out_dir / 'script.md'}")
    for w in info["storyboard_stats"].get("warnings", []):
        print(f"Check:  {w}")
    print(f"Time {info['total_s']:.0f}s (voice {info['timings']['voice_s']:.0f}s), "
          f"estimated model cost ${info['model_cost_usd_est']:.3f}")
    if args.check:
        import quality
        r = quality.check(out_dir)
        print(f"\nQuality: {r['status']}")
        for line in r["fail"] + r["issues"]:
            print(f"   - {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
