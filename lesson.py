"""
lesson.py - builds a text lesson on a topic, at a level and length, from the recordings.

How it works:
 1. Retrieve: search the index for the topic. Each hit also brings its neighbours (the
    minute before and after), because a single minute is rarely self-contained.
 2. Budget: if all recordings fit MAX_MATERIAL_WORDS, the model gets all of them (most
    relevant first); otherwise the best hits until the budget is used.
 3. Label: every excerpt gets a short label (S1, S2...). The model cites labels, not
    timestamps. Small models invent timestamps; labels we can check, and we map them
    back to exact meeting + time ourselves.
 4. Generate: the model must answer in a fixed JSON structure (Ollama enforces the schema).
    The same JSON later drives audio (script per section) and video option A (each
    section's sources = which clips to cut).
 5. Validate: JSON shape (pydantic), citations exist, length within +-15%. One automatic
    retry if the JSON is broken.
 6. Save lesson.json (everything, for evaluation) and lesson.md (for reading).

Usage:
  uv run lesson.py "budget planning" --minutes 5 --level beginner
  uv run lesson.py "budget planning" --minutes 5 --level expert --model <model id>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

import config
import index
import llm
import store

log = logging.getLogger("lesson")

WORDS_PER_MIN = 140    # comfortable reading/listening pace
TOP_K = 12             # search hits to start from
LENGTH_TOLERANCE = 0.15
LEVELS = {
    "beginner": "new to the topic: define every term, explain why things matter, use one simple example",
    "intermediate": "knows the basics: skip definitions of common terms, focus on how and when",
    "expert": "experienced: skip basics, focus on details, exceptions, decisions and trade-offs",
}


class Section(BaseModel):
    heading: str = Field(description="Short section heading")
    body: str = Field(description="Section text as plain prose paragraphs")
    sources: list[str] = Field(description="Labels of the excerpts used, e.g. ['S2', 'S5']")


class Lesson(BaseModel):
    title: str
    goal: str = Field(description="One sentence: what the learner will understand or be able to do")
    sections: list[Section]
    takeaways: list[str] = Field(description="3 short key takeaways")


class NotEnoughMaterial(Exception):
    pass


def _inline_refs(schema: dict) -> dict:
    """Replace pydantic's $ref/$defs with the definitions themselves (plainer for small models)."""
    defs = schema.get("$defs", {})

    def walk(node):
        """Copy the schema, replacing every $ref with the definition it points to."""
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


LESSON_SCHEMA = _inline_refs(Lesson.model_json_schema())


def material_budget(minutes: int) -> int:
    """How many transcript words the model gets. Claude (gateway) reads a lot cheaply; a small
    local model needs a short prompt, so there it scales with the lesson length."""
    if config.CHAT_BACKEND == "bifrost":
        return config.MAX_MATERIAL_WORDS
    return min(config.MAX_MATERIAL_WORDS, max(1200, minutes * WORDS_PER_MIN * 3))


def retrieve(topic: str, minutes: int, video_ids: set[str] | None = None) -> tuple[list[dict], float]:
    """Pick the excerpts for the lesson, in reading order. Returns (segments, best score).
    If everything in scope fits the budget, the model simply gets all of it (most relevant
    recording first); otherwise the best search hits plus the minute before and after."""
    hits = index.search(topic, top_k=TOP_K, video_ids=video_ids)
    if not hits or hits[0]["score"] < config.MIN_SCORE:
        best = "; ".join(f"{h['score']:.2f} {h['segment']['title']}" for h in hits[:3])
        raise NotEnoughMaterial(f"Recordings don't seem to cover '{topic}' (best matches: {best}). "
                                f"Try other wording, or lower MIN_SCORE in .env if the matches look right.")
    all_segs = {k: s for k, s in store.all_segments().items()
                if not video_ids or s["video_id"] in video_ids}
    budget = material_budget(minutes)
    video_rank: dict[str, int] = {}
    for h in hits:
        video_rank.setdefault(h["segment"]["video_id"], len(video_rank))

    def reading_order(segs) -> list[dict]:
        """Most relevant recording first, then chronological inside each recording."""
        return sorted(segs, key=lambda s: (video_rank.get(s["video_id"], 99), s["video_id"], s["idx"]))

    if sum(s["words"] for s in all_segs.values()) <= budget:
        return reading_order(all_segs.values()), hits[0]["score"]

    by_pos = {(s["video_id"], s["idx"]): s for s in all_segs.values()}
    chosen: dict[str, dict] = {}
    used = 0
    for h in hits:
        if h["score"] < config.MIN_SCORE * 0.9:
            break
        seg = h["segment"]
        for d in (0, -1, 1):   # the hit first, then the minute before and after
            s = by_pos.get((seg["video_id"], seg["idx"] + d))
            if s is None or s["id"] in chosen:
                continue
            if used + s["words"] > budget and chosen:
                break
            chosen[s["id"]] = s
            used += s["words"]
        if used >= budget:
            break
    return reading_order(chosen.values()), hits[0]["score"]


def recording_segments(video_ids: list[str], minutes: int) -> list[dict]:
    """All segments of the given recordings, in order, cut at the material budget."""
    segs = [s for s in store.all_segments().values() if s["video_id"] in video_ids]
    segs.sort(key=lambda s: (video_ids.index(s["video_id"]), s["idx"]))
    budget, used, out = material_budget(minutes), 0, []
    for s in segs:
        if used + s["words"] > budget and out:
            log.warning("Recording is longer than MAX_MATERIAL_WORDS=%s: using the first %s of %s segments",
                        budget, len(out), len(segs))
            break
        out.append(s)
        used += s["words"]
    return out


def format_material(segments: list[dict]) -> tuple[str, dict[str, dict]]:
    """Excerpts with labels S1..Sn. A recording's description and each slide's text are shown
    once, where they first appear, so the prompt doesn't repeat them."""
    labels, blocks, shown, introduced = {}, [], set(), set()
    for n, seg in enumerate(segments, 1):
        label = f"S{n}"
        labels[label] = seg
        head = f"[{label}] Recording \"{seg['title']}\", {store.fmt_ts(seg['start'])}-{store.fmt_ts(seg['end'])}"
        about = ""
        if seg.get("description") and seg["video_id"] not in introduced:
            introduced.add(seg["video_id"])
            about = f"\nAbout this recording: {seg['description']}"
        new_slides = [i for i in seg.get("slide_ids", []) if i not in shown]
        shown.update(new_slides)
        slide = f"\nOn screen: {seg['slide_text'][:400]}" if new_slides and seg.get("slide_text") else ""
        blocks.append(f"{head}{about}{slide}\n{seg['text']}")
    return "\n\n".join(blocks), labels


def build_prompt(topic: str, minutes: int, level: str, lang: str, material: str) -> str:
    """User message for a text lesson: the request (topic, level, length) and the excerpts."""
    target = minutes * WORDS_PER_MIN
    n_sections = max(2, min(6, round(minutes / 1.5)))
    return (
        f"Write a self-study lesson.\n"
        f"Topic: {topic}\n"
        f"Learner level: {level} ({LEVELS[level]})\n"
        f"Language: {lang}\n"
        f"Length: about {target} words in total. Use {n_sections} sections of about "
        f"{target // n_sections} words each, plus 3 takeaways.\n\n"
        f"SOURCE EXCERPTS\n\n{material}"
    )


def generate(prompt: str, system: str, model: str | None) -> Lesson:
    """Call the model; if the JSON is broken, show it the error and try once more."""
    messages = [{"role": "user", "content": prompt}]
    for attempt in (1, 2):
        text = llm.chat(messages, system=system, schema=LESSON_SCHEMA, step="compose_lesson",
                        model=model)
        try:
            return Lesson.model_validate(llm.parse_json(text))
        except (ValidationError, llm.LLMError) as e:
            if attempt == 2:
                raise llm.LLMError(f"Lesson JSON still invalid after a retry: {e}") from e
            log.warning("Invalid lesson JSON, retrying once: %s", str(e).splitlines()[0])
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That JSON was invalid: {e}. Return the corrected JSON only."}]
    raise AssertionError("unreachable")


def lesson_text(lesson: Lesson) -> str:
    """All text of a lesson in one string (for word counts and fact checks)."""
    parts = [lesson.title, lesson.goal] + [f"{s.heading} {s.body}" for s in lesson.sections] + lesson.takeaways
    return "\n".join(parts)


def render_markdown(lesson: Lesson, labels: dict[str, dict], meta: dict) -> str:
    """The lesson as readable Markdown, with its sources (recording + time) under each section."""
    out = [f"# {lesson.title}", "", f"**Goal:** {lesson.goal}", "",
           f"*Level: {meta['level']} · ~{meta['minutes']} min ({meta['stats']['words']} words)*", ""]
    for s in lesson.sections:
        out += [f"## {s.heading}", "", s.body.strip(), ""]
        refs = [labels[x] for x in dict.fromkeys(s.sources) if x in labels]
        if refs:
            out += ["*Sources: " + "; ".join(f"{r['title']} ({store.fmt_ts(r['start'])})" for r in refs) + "*", ""]
    out += ["## Key takeaways", ""] + [f"- {t}" for t in lesson.takeaways]
    out += ["", "---", f"*Generated {meta['created']} with {meta['model']} from "
            f"{len(labels)} excerpts of {len({r['video_id'] for r in labels.values()})} recordings.*"]
    return "\n".join(out)


def make_lesson(topic: str, minutes: int = 5, level: str = "beginner", lang: str = "English",
                model: str | None = None) -> dict:
    """The whole pipeline for one request. Returns the saved lesson record."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}")
    t0 = time.perf_counter()
    segments, best_score = retrieve(topic, minutes)
    material, labels = format_material(segments)
    system = store.load_skill("compose-lesson")
    lesson = generate(build_prompt(topic, minutes, level, lang, material), system, model)

    # --- checks the model can't be trusted with ---
    invalid = sorted({x for s in lesson.sections for x in s.sources if x not in labels})
    for s in lesson.sections:
        s.sources = [x for x in dict.fromkeys(s.sources) if x in labels]
    unsupported = [s.heading for s in lesson.sections if not s.sources]
    words = len(lesson_text(lesson).split())
    target = minutes * WORDS_PER_MIN
    stats = {
        "words": words, "target_words": target, "length_ratio": round(words / target, 2),
        "length_ok": abs(words / target - 1) <= LENGTH_TOLERANCE,
        "sections": len(lesson.sections), "excerpts_given": len(labels),
        "excerpts_cited": len({x for s in lesson.sections for x in s.sources}),
        "invalid_citations": invalid, "sections_without_sources": unsupported,
        "best_search_score": best_score, "seconds": round(time.perf_counter() - t0, 1),
    }
    for warning, active in [(f"length {words} words vs target {target}", not stats["length_ok"]),
                            (f"invented citations removed: {invalid}", bool(invalid)),
                            (f"sections without sources: {unsupported}", bool(unsupported))]:
        if active:
            log.warning(warning)

    created = datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {
        "lesson_id": f"{created}-{store.slugify(topic, 40)}-{minutes}m-{level}",
        "created": created, "topic": topic, "minutes": minutes, "level": level, "language": lang,
        "model": model or config.CHAT_MODEL, "stats": stats, "lesson": lesson.model_dump(),
        "sources": {k: {"segment_id": v["id"], "video_id": v["video_id"], "title": v["title"],
                        "start": v["start"], "end": v["end"]} for k, v in labels.items()},
    }
    path = store.save_lesson(record["lesson_id"], record, render_markdown(lesson, labels, record))
    record["path"] = str(path)
    log.info("Saved %s (%s words, %.0fs)", path, words, stats["seconds"])
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic")
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument("--level", choices=list(LEVELS), default="beginner")
    ap.add_argument("--lang", default="English")
    ap.add_argument("--model", help="override CHAT_MODEL from .env")
    args = ap.parse_args()
    config.setup_logging()
    try:
        record = make_lesson(args.topic, args.minutes, args.level, args.lang, args.model)
    except NotEnoughMaterial as e:
        log.error(str(e))
        return 2
    except (RuntimeError, ValueError, OSError) as e:   # model, index or file problems
        log.error("%s", e)
        return 1
    print(open(f"{record['path']}/lesson.md", encoding="utf-8").read())
    print("\nStats:", json.dumps(record["stats"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
