"""
quality.py - automatic checks of a finished video lesson, so you only need to look closely at
the ones that need it.

Checks without a model call (free):
  length      narration words vs target (within 25% is fine)
  sync        video and audio tracks of the mp4 have the same length
  sources     invented source labels removed, scenes without sources
  read-aloud  a slide scene whose narration repeats the slide text word for word
  on-screen   key points / steps longer than 8 words, headings longer than 10 words
  slides      the same slide shown twice
  voice       unusual words (acronyms, CamelCase, words with digits) that are not in
              pronunciation.csv: listen to them, add a row if the voice says them wrongly
Optional review by the chat model (--review, about $0.01 per video): every scene is compared
with the transcript excerpts it cites; scores 1-5 for faithful / level fit / clarity, plus a
list of problems.

Result: quality.json in the video folder, with status OK, CHECK (look at it), FAIL, or PLAN
(storyboard only, not rendered yet). A review is kept until the storyboard changes.

Usage:
  uv run quality.py data/output/<folder>             # one video
  uv run quality.py data/output/<folder> --review    # plus the model review
  uv run quality.py --all [--review]                 # every video folder
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import config
import lesson
import llm
import media
import storyboard
import store
import tts

log = logging.getLogger("quality")

READ_ALOUD_LIMIT = 0.5     # share of the slide's 4-word phrases repeated in the narration
LENGTH_TOLERANCE = 0.25
MAX_ITEM_WORDS = 8
MAX_HEADING_WORDS = 10
SYNC_TOLERANCE_S = 0.5
COMMON_ACRONYMS = {"OK", "AI", "IT", "HR", "US", "UK", "EU", "PDF", "FAQ", "ID", "TV", "PC", "I"}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "integer", "description": "1-5: every statement is supported by the excerpts"},
        "level_fit": {"type": "integer", "description": "1-5: suits the learner level"},
        "clarity": {"type": "integer", "description": "1-5: easy to follow when heard once"},
        "issues": {"type": "array", "items": {"type": "object", "properties": {
            "scene": {"type": "integer"}, "problem": {"type": "string"}},
            "required": ["scene", "problem"]}},
    },
    "required": ["faithful", "level_fit", "clarity", "issues"],
}
REVIEW_PROMPT = """You review a narrated video lesson made from internal training recordings.
For each scene you get the narration, what the screen shows, and the transcript excerpts it
was based on. Score from 1 (bad) to 5 (very good):
- faithful: every statement is supported by the excerpts (no invented facts, numbers or names)
- level_fit: suits a {level} learner
- clarity: easy to follow when heard once
List concrete problems with the scene number (empty list if none). Be strict but fair."""


def _words(text: str) -> list[str]:
    """Lowercase words (letters and digits) of a text."""
    return re.findall(r"[a-z0-9]+", text.lower())


def read_aloud_ratio(narration: str, slide_text: str, n: int = 4) -> float:
    """Share of the slide's n-word phrases that appear unchanged in the narration (0 to 1).
    High values mean the voice reads the slide out instead of explaining it."""
    slide, spoken = _words(slide_text), _words(narration)
    if len(slide) < 2 * n:
        return 0.0
    slide_grams = {tuple(slide[i:i + n]) for i in range(len(slide) - n + 1)}
    spoken_grams = {tuple(spoken[i:i + n]) for i in range(len(spoken) - n + 1)}
    return len(slide_grams & spoken_grams) / len(slide_grams)


def unusual_words(text: str, known: set[str]) -> list[str]:
    """Words a voice may say wrongly: acronyms (KPI), CamelCase (PowerBI), words with digits (Q3)."""
    found = re.findall(r"\b(?:[A-Z]{2,}s?|[A-Z][a-z]+[A-Z][A-Za-z]*|[A-Za-z]*\d[A-Za-z\d]*)\b", text)
    return sorted({w for w in found if w not in known and w not in COMMON_ACRONYMS and not w.isdigit()})


def check(out_dir: Path, review: bool = False) -> dict:
    """Run all checks on one video folder, save quality.json, return the result."""
    sb = storyboard.load(out_dir)
    issues: list[str] = []
    fail: list[str] = []
    metrics: dict = {}
    scenes = sb["scenes"]

    # length
    words = sum(len(s["narration"].split()) for s in scenes)
    target = sb.get("stats", {}).get("target_words") or int(sb["minutes"] * storyboard.SPOKEN_WPM)
    metrics.update(words=words, target_words=target, length_ratio=round(words / max(target, 1), 2))
    if abs(words / max(target, 1) - 1) > LENGTH_TOLERANCE:
        issues.append(f"narration is {words} words, target {target}")

    # warnings from the storyboard step (invented sources, scenes without sources ...)
    issues += sb.get("stats", {}).get("warnings", [])

    # scene by scene
    seen_slides: set[str] = set()
    for i, sc in enumerate(scenes, 1):
        if sc["visual"] == "slide":
            ratio = read_aloud_ratio(sc["narration"], sc.get("slide_text", ""))
            if ratio >= READ_ALOUD_LIMIT:
                issues.append(f"scene {i}: narration reads the slide aloud ({ratio:.0%} of its phrases)")
            if sc.get("slide") in seen_slides:
                issues.append(f"scene {i}: slide {sc['slide']} shown twice")
            seen_slides.add(sc.get("slide", ""))
            if sc.get("slide_image") and not Path(sc["slide_image"]).exists():
                issues.append(f"scene {i}: slide image missing ({sc['slide_image']})")
        long_items = [x for x in sc.get("items", []) if len(x.split()) > MAX_ITEM_WORDS]
        if long_items:
            issues.append(f"scene {i}: on-screen text too long: {long_items[0][:60]}")
        if len(sc.get("heading", "").split()) > MAX_HEADING_WORDS:
            issues.append(f"scene {i}: heading too long")

    # voice
    known = set(tts.load_pronunciations())
    odd = unusual_words(" ".join(s["narration"] for s in scenes), known)
    metrics["words_to_listen_to"] = odd
    if odd:
        issues.append(f"listen to how these are spoken (add to pronunciation.csv if wrong): {', '.join(odd[:12])}")

    # files and sync
    video = out_dir / "lesson_video.mp4"
    planned_only = not video.exists()
    if planned_only:
        issues.append("not rendered yet (storyboard only): uv run make_video.py --rerender <folder>")
    else:
        try:
            d = media.stream_durations(video)
            metrics["video_s"], metrics["audio_s"] = round(d.get("video", 0), 1), round(d.get("audio", 0), 1)
            gap = abs(d.get("video", 0) - d.get("audio", 0))
            if not d.get("video") or not d.get("audio"):
                fail.append("video or audio track missing in the mp4")
            elif gap > 2:
                fail.append(f"video and audio differ by {gap:.1f}s")
            elif gap > SYNC_TOLERANCE_S:
                issues.append(f"video and audio differ by {gap:.1f}s")
        except media.MediaError as e:
            fail.append(f"mp4 unreadable: {e}")

    result = {"fail": fail, "issues": issues, "metrics": metrics}
    previous = store.read_json(out_dir / "quality.json", {}) or {}
    if review:
        try:
            result["review"] = model_review(sb)
        except llm.LLMError as e:
            result["review_error"] = str(e)[:300]
            log.warning("Review failed: %s", e)
    elif previous.get("review") and previous.get("storyboard_created") == sb.get("created"):
        result["review"] = previous["review"]   # keep the review of the same storyboard
    result["storyboard_created"] = sb.get("created")
    result["status"] = _status(result, planned_only)
    store.write_json(out_dir / "quality.json", result)
    return result


def _status(result: dict, planned_only: bool) -> str:
    """FAIL if something is broken or the review finds invented content; PLAN if not rendered
    yet; CHECK if anything is worth a look; else OK."""
    review = result.get("review")
    if review and review["faithful"] <= 2:
        result["fail"].append(f"review: faithful {review['faithful']}/5")
    if result["fail"]:
        return "FAIL"
    if planned_only:
        return "PLAN"
    if result["issues"] or (review and min(review["faithful"], review["level_fit"], review["clarity"]) <= 3):
        return "CHECK"
    return "OK"


def model_review(sb: dict) -> dict:
    """Ask the review model to score the storyboard against the excerpts each scene cites."""
    segments = store.all_segments()
    blocks = []
    for i, sc in enumerate(sb["scenes"], 1):
        screen = sc["visual"] + (f" ({sc['heading']})" if sc.get("heading") else "")
        if sc.get("items"):
            screen += ": " + "; ".join(sc["items"])
        cited = []
        for label in sc.get("sources", []):
            seg = segments.get(sb.get("sources", {}).get(label, {}).get("segment_id", ""))
            if seg:
                cited.append(f"[{label}] {seg['text'][:1500]}")
        blocks.append(f"SCENE {i}\nScreen: {screen}\nNarration: {sc['narration']}\n"
                      f"Excerpts:\n" + ("\n".join(cited) or "(none cited)"))
    prompt = f"Lesson: {sb['title']}\nGoal: {sb['goal']}\n\n" + "\n\n".join(blocks)
    level = sb.get("level", "beginner")
    data = llm.chat_json(prompt, REVIEW_SCHEMA, system=REVIEW_PROMPT.format(level=f"{level} ({lesson.LEVELS.get(level, '')})"),
                         step="review", model=config.REVIEW_MODEL or None, max_tokens=1500, temperature=0.0)
    for key in ("faithful", "level_fit", "clarity"):
        data[key] = max(1, min(5, int(data.get(key, 0) or 0)))
    data["issues"] = [i for i in data.get("issues", []) if isinstance(i, dict)][:15]
    return data


def video_folders() -> list[Path]:
    """Every output folder that holds a video storyboard, newest first."""
    if not config.OUTPUT_DIR.exists():
        return []
    return sorted((p.parent for p in config.OUTPUT_DIR.glob("*/storyboard.json")), reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", type=Path, help="a video output folder")
    ap.add_argument("--all", action="store_true", help="check every video folder")
    ap.add_argument("--review", action="store_true", help="also ask the model to review (~$0.01 each)")
    args = ap.parse_args()
    config.setup_logging()
    folders = video_folders() if args.all else ([args.folder] if args.folder else [])
    if not folders:
        ap.error("give a folder or --all")
    worst = 0
    for folder in folders:
        try:
            r = check(folder, args.review)
        except (RuntimeError, ValueError, OSError) as e:
            log.error("%s: %s", folder.name, e)
            worst = 2
            continue
        print(f"\n[{r['status']}] {folder.name}")
        for line in r["fail"] + r["issues"]:
            print(f"   - {line}")
        if "review" in r:
            rv = r["review"]
            print(f"   review: faithful {rv['faithful']}/5, level {rv['level_fit']}/5, clarity {rv['clarity']}/5")
            for i in rv["issues"]:
                print(f"     scene {i.get('scene')}: {i.get('problem')}")
        worst = max(worst, {"OK": 0, "PLAN": 0, "CHECK": 1, "FAIL": 2}[r["status"]])
    print("\nReport for all videos: uv run report.py")
    return 1 if worst == 2 else 0


if __name__ == "__main__":
    sys.exit(main())
