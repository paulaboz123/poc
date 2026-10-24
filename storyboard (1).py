"""
storyboard.py - asks the chat model for the plan of a narrated video lesson: for each scene,
what the voice says and what the screen shows.

Input:  transcript excerpts (labelled S1, S2 ...) and the slides that were on screen while
        they were spoken (P1, P2 ...), plus the request (topic, minutes, level).
Output: storyboard.json (and script.md for reading). The storyboard is plain JSON on purpose:
        you can edit narration or on-screen text by hand and re-render the video with
        make_video.py --rerender, without another model call.

By default the video shows only NEW slides drawn from the content (ORIGINAL_SLIDES=no in .env):
the original slides' text and the descriptions of their diagrams are given to the model as
knowledge, and it redesigns them as key points, steps, comparisons, tables or concept cards.

The model's answer is checked and repaired, because a video must not break on one bad field:
- an original slide when they are switched off, or a slide label that does not exist
  -> the scene shows its heading and key points instead
- a comparison, table or concept card without enough content -> key points
- a 'steps' scene with fewer than 2 steps -> bullets; empty scenes are dropped
- the first scene is always a title scene
- invented source labels are removed, and the word count is compared with the target
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

import config
import lesson
import llm
import store

log = logging.getLogger("storyboard")

SPOKEN_WPM = 150        # Kokoro at speed 1.0 speaks ~150-160 words per minute
SCENE_SECONDS = 20      # average scene length; ~3 scenes per minute keeps the picture moving
MAX_ITEMS = 6


VISUALS = ("title", "bullets", "steps", "comparison", "table", "concept", "slide")


class Column(BaseModel):
    heading: str = Field(description="Column heading, 1-4 words")
    items: list[str] = Field(description="2-4 short points, at most 6 words each")


class Scene(BaseModel):
    visual: Literal["title", "bullets", "steps", "comparison", "table", "concept", "slide"] = Field(
        description="What the screen shows: title | bullets (key points) | steps (a process in order) | "
                    "comparison (2-3 columns side by side) | table (a small grid) | concept (a term and its "
                    "definition) | slide (an original slide, only if offered)")
    heading: str = Field(default="", description="Short on-screen heading. Title scene: the lesson title. "
                                                 "Concept: the term")
    items: list[str] = Field(default_factory=list,
                             description="bullets/steps: 2-5 short phrases, at most 8 words each. "
                                         "concept: [definition in at most 20 words, optional short example]")
    columns: list[Column] = Field(default_factory=list, description="comparison only: 2-3 columns")
    table_header: list[str] = Field(default_factory=list, description="table only: 2-4 column names")
    table_rows: list[list[str]] = Field(default_factory=list,
                                        description="table only: 2-6 rows, one short cell per column")
    slide: str = Field(default="", description="For visual=slide: the slide label to show, e.g. P2")
    narration: str = Field(description="What the voice says during this scene: 25-70 words of spoken English")
    sources: list[str] = Field(default_factory=list, description="S-labels the narration is based on, e.g. ['S2']")


class Storyboard(BaseModel):
    title: str = Field(description="Lesson title, specific to the topic")
    goal: str = Field(description="One sentence: what the learner can do after watching")
    scenes: list[Scene]


def _clean_schema(schema: dict) -> dict:
    """Inline $refs (simpler for models) and drop 'default'/'title' keys some validators reject."""
    schema = lesson._inline_refs(schema)

    def walk(node):
        """Copy the schema without 'default' keys and without metadata 'title' strings."""
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items() if k not in ("default", "title")
                    or (k == "title" and isinstance(v, dict))}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


SCHEMA = _clean_schema(Storyboard.model_json_schema())


def schema_for_request() -> dict:
    """The schema sent to the model; without the 'slide' visual when original slides are off."""
    if config.ORIGINAL_SLIDES:
        return SCHEMA
    schema = json.loads(json.dumps(SCHEMA))
    visual = schema["properties"]["scenes"]["items"]["properties"]["visual"]
    visual["enum"] = [v for v in visual["enum"] if v != "slide"]
    visual["description"] = visual["description"].replace(" | slide (an original slide, only if offered)", "")
    return schema


# --- Material -----------------------------------------------------------------------------
def build_material(segments: list[dict]) -> tuple[str, dict[str, dict], dict[str, dict]]:
    """Prompt text with S- and P-labels, plus label -> segment and label -> slide lookups."""
    slides_by_video: dict[str, dict[str, dict]] = {}
    p_labels: dict[str, dict] = {}
    slide_label: dict[str, str] = {}
    for seg in segments:   # slides in order of first appearance
        vid = seg["video_id"]
        if vid not in slides_by_video:
            slides_by_video[vid] = {s["id"]: s for s in store.load_slides(vid)}
        for sid in seg.get("slide_ids", []):
            sl = slides_by_video[vid].get(sid)
            if sl and sl.get("is_slide") and sid not in slide_label:
                label = f"P{len(p_labels) + 1}"
                slide_label[sid] = label
                p_labels[label] = {**sl, "title": seg["title"],
                                   "image": str(store.video_dir(vid) / sl["file"])}

    slide_lines = []
    for label, sl in p_labels.items():
        text = " / ".join(x.strip() for x in sl.get("text", "").splitlines() if x.strip())[:350]
        visual = f"\n    What it shows: {sl['visual'][:400]}" if sl.get("visual") else ""
        slide_lines.append(f"[{label}] slide from \"{sl['title']}\" at {store.fmt_ts(sl['start'])}: {text}{visual}")

    s_labels: dict[str, dict] = {}
    blocks, introduced = [], set()
    for n, seg in enumerate(segments, 1):
        label = f"S{n}"
        s_labels[label] = seg
        head = f"[{label}] Recording \"{seg['title']}\", {store.fmt_ts(seg['start'])}-{store.fmt_ts(seg['end'])}"
        if seg.get("description") and seg["video_id"] not in introduced:
            introduced.add(seg["video_id"])
            head += f"\nAbout this recording: {seg['description']}"
        on_screen = [slide_label[s] for s in seg.get("slide_ids", []) if s in slide_label]
        if on_screen:
            head += f"\nSlides on screen: {', '.join(dict.fromkeys(on_screen))}"
        blocks.append(f"{head}\n{seg['text']}")

    if config.ORIGINAL_SLIDES:
        slides_head = "SLIDES YOU MAY SHOW"
        empty = "(none: design new slides)"
    else:
        slides_head = ("CONTENT OF THE ORIGINAL SLIDES (knowledge only: never show or copy these slides; "
                       "redesign their content as new slides)")
        empty = "(none)"
    text = (slides_head + "\n" + ("\n".join(slide_lines) or empty)
            + "\n\nTRANSCRIPT EXCERPTS\n\n" + "\n\n".join(blocks))
    return text, s_labels, p_labels


def build_prompt(topic: str, minutes: float, level: str, material: str, summary_of: str = "",
                 wishes: str = "") -> str:
    """User message for the storyboard: task, topic, level, word target and scene count,
    then the material."""
    target = int(minutes * SPOKEN_WPM)
    n_scenes = max(4, round(minutes * 60 / SCENE_SECONDS))
    task = (f"Make the storyboard of a narrated video lesson that teaches the content of the "
            f"recording(s) {summary_of}." if summary_of else
            "Make the storyboard of a narrated video lesson.")
    extra = f"Wishes from the person who asked (follow them within the rules): {wishes}\n" if wishes else ""
    return (f"{task}\nTopic: {topic}\nLearner level: {level} ({lesson.LEVELS[level]})\n"
            f"Length: about {target} spoken words in total (~{minutes:g} min), in about {n_scenes} scenes.\n"
            f"{extra}\n{material}")


# --- Generation -----------------------------------------------------------------------------
MAX_SCREEN_WORDS = {"items": 8, "columns": 6, "table": 5, "heading": 8}


def long_screen_texts(sb: Storyboard) -> list[str]:
    """On-screen texts longer than the limits: they make a slide crowded and get read word by word."""
    found = []
    for i, sc in enumerate(sb.scenes, 1):
        limit = 20 if sc.visual == "concept" else MAX_SCREEN_WORDS["items"]
        found += [f"scene {i}: {x}" for x in sc.items if len(x.split()) > limit]
        found += [f"scene {i}: {x}" for col in sc.columns for x in col.items
                  if len(x.split()) > MAX_SCREEN_WORDS["columns"]]
        found += [f"scene {i}: {x}" for row in sc.table_rows for x in row if len(x.split()) > MAX_SCREEN_WORDS["table"]]
        if sc.visual != "title" and len(sc.heading.split()) > MAX_SCREEN_WORDS["heading"]:
            found.append(f"scene {i}: {sc.heading}")
    return found


def generate(prompt: str, system: str, model: str | None) -> Storyboard:
    """Call the model; if the JSON is broken, show it the error and try once more. If on-screen
    texts are too long, ask once for shorter ones (the narration stays the same)."""
    messages = [{"role": "user", "content": prompt}]
    fixes_left = {"json": 1, "short": 1}
    while True:
        text = llm.chat(messages, system=system, schema=schema_for_request(), step="storyboard", model=model,
                        temperature=0.4)
        try:
            sb = Storyboard.model_validate(llm.parse_json(text))
        except (ValidationError, llm.LLMError) as e:
            if not fixes_left["json"]:
                raise llm.LLMError(f"Storyboard JSON still invalid after a retry: {e}") from e
            fixes_left["json"] -= 1
            log.warning("Invalid storyboard JSON, retrying once: %s", str(e).splitlines()[0])
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That JSON was invalid: {e}. Return the corrected JSON only."}]
            continue
        too_long = long_screen_texts(sb)
        if not too_long or not fixes_left["short"]:
            return sb
        fixes_left["short"] -= 1
        log.info("Asking for shorter on-screen texts (%s too long)", len(too_long))
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": (
                         "These on-screen texts are too long:\n" + "\n".join(f"- {t}" for t in too_long[:20]) +
                         "\n\nReturn the same JSON with only those texts shortened: key points and steps at most "
                         "8 words, comparison points 6, table cells 5, headings 8, a concept definition 20. "
                         "Keep everything else, including the narration, exactly the same.")}]


def _short(text: str) -> str:
    """One line of on-screen text: single spaces, no trailing full stop."""
    return " ".join(text.replace("\n", " ").split()).strip(" .;")


def repair(sb: Storyboard, s_labels: dict, p_labels: dict, minutes: float) -> tuple[list[dict], dict]:
    """Turn the model's storyboard into safe scene dicts. Returns (scenes, stats)."""
    warnings: list[str] = []
    invalid_refs: set[str] = set()
    scenes: list[dict] = []
    used_slides: set[str] = set()
    for i, sc in enumerate(sb.scenes):
        narration = " ".join(sc.narration.split())
        if not narration:
            warnings.append(f"scene {i + 1}: empty narration, dropped")
            continue
        items = [_short(x) for x in sc.items if _short(x)][:MAX_ITEMS]
        visual = sc.visual
        extra: dict = {}
        ref = sc.slide.strip().strip("[]").upper()
        if visual == "slide" and not config.ORIGINAL_SLIDES:
            visual = "bullets"
        elif visual == "slide" and ref not in p_labels:
            warnings.append(f"scene {i + 1}: unknown slide {sc.slide!r}, showing key points instead")
            visual = "bullets"
        if visual == "steps" and len(items) < 2:
            visual = "bullets"
        if visual == "title" and scenes:   # a title in the middle becomes a section card
            visual = "bullets"
        if visual == "comparison":
            cols = [{"heading": _short(c.heading), "items": [_short(x) for x in c.items if _short(x)][:5]}
                    for c in sc.columns if _short(c.heading)][:3]
            if len(cols) >= 2 and all(c["items"] for c in cols):
                extra["columns"] = cols
            else:
                visual = "bullets"
                items = items or [f"{c['heading']}: {', '.join(c['items'])}" for c in cols][:MAX_ITEMS]
        if visual == "table":
            header = [_short(h) for h in sc.table_header if _short(h)][:4]
            rows = [[_short(c) for c in r][:len(header)] for r in sc.table_rows if any(_short(c) for c in r)][:6]
            rows = [r + [""] * (len(header) - len(r)) for r in rows]
            if len(header) >= 2 and len(rows) >= 2:
                extra.update(table_header=header, table_rows=rows)
            else:
                visual = "bullets"
                items = items or [" - ".join(x for x in r if x) for r in rows][:MAX_ITEMS]
        if visual == "concept" and not (_short(sc.heading) and items):
            visual = "bullets"
        invalid_refs |= {x for x in sc.sources if x not in s_labels}
        scene = {"visual": visual, "heading": _short(sc.heading), "items": items, "narration": narration,
                 "sources": [x for x in dict.fromkeys(sc.sources) if x in s_labels], **extra}
        if visual == "slide":
            sl = p_labels[ref]
            if ref in used_slides:
                warnings.append(f"scene {i + 1}: slide {ref} shown twice")
            used_slides.add(ref)
            scene.update({"slide": ref, "slide_image": sl["image"], "slide_text": sl.get("text", "")[:800],
                          "caption": f"From: {sl['title']} · {store.fmt_ts(sl['start'])}"})
        scenes.append(scene)
    if not scenes:
        raise llm.LLMError("The storyboard has no usable scenes")
    if scenes[0]["visual"] != "title":
        scenes.insert(0, {"visual": "title", "heading": sb.title, "items": [], "sources": [],
                          "narration": f"{sb.title}. {sb.goal}"})
    scenes[0]["heading"] = scenes[0]["heading"] or sb.title

    words = sum(len(s["narration"].split()) for s in scenes)
    target = int(minutes * SPOKEN_WPM)
    if invalid_refs:
        warnings.append(f"invented source labels removed: {sorted(invalid_refs)}")
    uncited = [i + 1 for i, s in enumerate(scenes) if not s["sources"] and s["visual"] != "title"]
    if uncited:
        warnings.append(f"scenes without sources: {uncited}")
    if abs(words / target - 1) > 0.25:
        warnings.append(f"narration is {words} words, target {target}")
    for w in warnings:
        log.warning(w)
    stats = {"scenes": len(scenes), "words": words, "target_words": target,
             "length_ratio": round(words / target, 2), "est_minutes": round(words / SPOKEN_WPM, 1),
             "slides_offered": len(p_labels), "slides_used": len(used_slides),
             "excerpts_given": len(s_labels),
             "excerpts_cited": len({x for s in scenes for x in s["sources"]}), "warnings": warnings}
    return scenes, stats


# --- Main entry -------------------------------------------------------------------------------
def make(topic: str, minutes: float, level: str, segments: list[dict], *, model: str | None = None,
         summary_of: str = "", wishes: str = "") -> dict:
    """Material -> model -> repaired storyboard dict (not saved yet)."""
    if level not in lesson.LEVELS:
        raise ValueError(f"level must be one of {list(lesson.LEVELS)}")
    material, s_labels, p_labels = build_material(segments)
    log.info("Material: %s excerpts (%s words), %s slides", len(s_labels),
             sum(s["words"] for s in segments), len(p_labels))
    sb = generate(build_prompt(topic, minutes, level, material, summary_of, wishes),
                  store.load_skill("compose-video"), model)
    scenes, stats = repair(sb, s_labels, p_labels, minutes)
    return {
        "created": datetime.now().strftime("%Y%m%d-%H%M%S"), "topic": topic, "minutes": minutes,
        "level": level, "model": model or config.CHAT_MODEL, "title": sb.title, "goal": sb.goal,
        "scenes": scenes, "stats": stats,
        "sources": {k: {"segment_id": v["id"], "video_id": v["video_id"], "title": v["title"],
                        "start": v["start"], "end": v["end"]} for k, v in s_labels.items()},
    }


def script_markdown(sb: dict) -> str:
    """Human-readable version of the storyboard, for review."""
    out = [f"# {sb['title']}", "", f"**Goal:** {sb['goal']}", "",
           f"*{sb['level']} · target ~{sb['minutes']:g} min · {sb['stats']['words']} words · "
           f"{len(sb['scenes'])} scenes*", ""]
    for i, sc in enumerate(sb["scenes"], 1):
        screen = sc["visual"] + (f" {sc['slide']}" if sc.get("slide") else "")
        out += [f"## Scene {i} · {screen}" + (f" · {sc['heading']}" if sc["heading"] else "")]
        if sc["items"]:
            out += [""] + [f"- {x}" for x in sc["items"]]
        for col in sc.get("columns", []):
            out += ["", f"**{col['heading']}**"] + [f"- {x}" for x in col["items"]]
        if sc.get("table_header"):
            out += ["", "| " + " | ".join(sc["table_header"]) + " |",
                    "|" + " --- |" * len(sc["table_header"])]
            out += ["| " + " | ".join(r) + " |" for r in sc["table_rows"]]
        refs = "; ".join(f"{sb['sources'][x]['title']} ({store.fmt_ts(sb['sources'][x]['start'])})"
                         for x in sc["sources"] if x in sb["sources"])
        out += ["", f"> {sc['narration']}", ""] + ([f"*Sources: {refs}*", ""] if refs else [])
    return "\n".join(out)


def save(sb: dict, out_dir: Path) -> Path:
    """Write storyboard.json and script.md into the output folder. Returns the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    store.write_json(out_dir / "storyboard.json", sb)
    (out_dir / "script.md").write_text(script_markdown(sb), encoding="utf-8")
    return out_dir / "storyboard.json"


def load(out_dir: Path) -> dict:
    """Read storyboard.json from an output folder (e.g. after editing it by hand)."""
    path = out_dir / "storyboard.json"
    if not path.exists():
        raise FileNotFoundError(f"No storyboard.json in {out_dir}")
    return json.loads(path.read_text(encoding="utf-8"))
