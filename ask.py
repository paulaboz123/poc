"""
ask.py - make a lesson from a request in your own words, the way users will ask for it later.

  uv run ask.py "I need a 2-minute video on joining data for beginners, as a presentation with audio"
  uv run ask.py "Hej, chcę 3 minuty o typach złączeń, poziom średni, z tabelą porównawczą"
  uv run ask.py "Summarise the recording about data cleaning in 5 minutes as a text lesson"

How it works:
 1. The chat model reads the request together with the list of available recordings and
    turns it into settings: format (video / audio / text), topic, recordings, minutes, level,
    and any wishes about the look ("with a comparison table", "more examples").
 2. The settings are printed, so you see how the request was understood.
 3. The usual pipeline runs: make_video.py (video + audio + script) or lesson.py (text).
Requests can be in any language; the lesson itself is written in English.

Options:  --plan-only  (video: storyboard only, check script.md first)   --check  (quality checks)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
import lesson
import llm
import make_video
import store

log = logging.getLogger("ask")

REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "format": {"type": "string", "enum": ["video", "audio", "text"],
                   "description": "video = narrated presentation with visuals and audio; audio = listening only; "
                                  "text = written lesson"},
        "topic": {"type": "string", "description": "What to teach, in English, a short phrase"},
        "recordings": {"type": "array", "items": {"type": "string"},
                       "description": "ids of the recordings the request points to; empty = search all"},
        "minutes": {"type": "number", "description": "Target length in minutes"},
        "level": {"type": "string", "enum": ["beginner", "intermediate", "expert"]},
        "wishes": {"type": "string", "description": "Wishes about content or look, in English; empty if none"},
    },
    "required": ["format", "topic", "recordings", "minutes", "level", "wishes"],
}
SYSTEM = """You turn a learner's request for a lesson into settings. The request can be in any
language; write topic and wishes in English. Rules:
- format: video if they ask for a video, presentation, slides or visuals; audio if they ask for
  something to listen to (podcast, audio); text if they ask for something to read. Default: video.
- recordings: only if the request clearly points to specific recordings (by number or by their
  subject); use the ids from the list. Otherwise an empty list, and the topic is searched in all.
- minutes: as asked; default 2 for video and audio, 5 for text; between 1 and 10.
- level: beginner unless they say they know the basics (intermediate) or are experienced (expert).
- wishes: anything else they ask for about content or look (e.g. "use a comparison table",
  "include an example"), otherwise empty."""


def understand(request: str) -> dict:
    """Ask the chat model to turn a request in plain words into settings."""
    rows = store.available_recordings()
    if not rows:
        raise ValueError("No processed recordings yet: set include=yes in data/manifest.csv and run pipeline.py.")
    listing = "\n".join(f"- id {r['video_id']}: {r['title']}" + (f" ({r['description']})" if r.get("description") else "")
                        for r in rows)
    prompt = f"Available recordings:\n{listing}\n\nRequest:\n{request}"
    data = llm.chat_json(prompt, REQUEST_SCHEMA, system=SYSTEM, step="understand_request",
                         max_tokens=500, temperature=0.0)
    known = {r["video_id"] for r in rows}
    data["recordings"] = [x for x in data.get("recordings", []) if x in known]
    data["minutes"] = max(1.0, min(10.0, float(data.get("minutes") or 2)))
    if data.get("level") not in lesson.LEVELS:
        data["level"] = "beginner"
    if data.get("format") not in ("video", "audio", "text"):
        data["format"] = "video"
    data["topic"] = (data.get("topic") or "").strip() or request[:80]
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", help="the request in your own words (in quotes)")
    ap.add_argument("--plan-only", action="store_true", help="video: stop after the storyboard")
    ap.add_argument("--check", action="store_true", help="video: run the quality checks afterwards")
    ap.add_argument("--model", help="override CHAT_MODEL from .env")
    args = ap.parse_args()
    config.setup_logging()
    try:
        s = understand(args.request)
        titles = {r["video_id"]: r["title"] for r in store.available_recordings()}
        print("\nUnderstood as:")
        print(f"  format:     {s['format']}")
        print(f"  topic:      {s['topic']}")
        print(f"  recordings: {', '.join(titles[v] for v in s['recordings']) or 'all (searched by topic)'}")
        print(f"  length:     {s['minutes']:g} min, level {s['level']}")
        if s.get("wishes"):
            print(f"  wishes:     {s['wishes']}")
        print()
        if s["format"] == "text":
            record = lesson.make_lesson(s["topic"], int(round(s["minutes"])), s["level"], model=args.model)
            print(f"\nText lesson: {Path(record['path']) / 'lesson.md'}")
            return 0
        out_dir, info = make_video.make(s["topic"], ",".join(s["recordings"]) or None, s["minutes"], s["level"],
                                        args.model, args.plan_only, wishes=s.get("wishes", ""))
    except lesson.NotEnoughMaterial as e:
        log.error("%s", e)
        return 2
    except (RuntimeError, ValueError, OSError) as e:
        log.error("%s", e)
        return 1

    if args.plan_only:
        print(f"Storyboard ready: {out_dir / 'script.md'}")
        print(f"Then: uv run make_video.py --rerender \"{out_dir}\"")
        return 0
    if s["format"] == "audio":
        print(f"Audio lesson: {info['audio']}   (the video version is next to it: {info['video']})")
    else:
        print(f"Video: {info['video']}\nAudio: {info['audio']}")
    print(f"Script: {out_dir / 'script.md'}\nEstimated model cost: ${llm.spent_this_run():.3f}")
    if args.check:
        import quality
        r = quality.check(out_dir)
        print(f"\nQuality: {r['status']}")
        for line in r["fail"] + r["issues"]:
            print(f"   - {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
