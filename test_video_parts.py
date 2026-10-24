"""Storyboard repair, drawing scenes, pronunciation and the quality helpers."""
from __future__ import annotations

import json

from PIL import Image

import config
import quality
import render
import storyboard
import tts


def test_storyboard_is_repaired_instead_of_breaking_the_video():
    sb = storyboard.Storyboard.model_validate({"title": "T", "goal": "G", "scenes": [
        {"visual": "bullets", "heading": "Intro", "items": ["a"], "narration": "Hello.", "sources": ["S1"]},
        {"visual": "slide", "slide": "[p9]", "heading": "X", "narration": "Talk.", "sources": ["S2", "S8"]},
        {"visual": "steps", "heading": "One step", "items": ["x"], "narration": "Steps.", "sources": []},
        {"visual": "bullets", "heading": "Empty", "narration": "  ", "sources": []}]})
    scenes, stats = storyboard.repair(sb, {"S1": {}, "S2": {}}, {}, 1)
    assert [s["visual"] for s in scenes] == ["title", "bullets", "bullets", "bullets"]   # title added, bad slide -> bullets
    assert scenes[2]["sources"] == ["S2"]                                                # invented S8 removed
    assert any("S8" in w for w in stats["warnings"]) and any("empty narration" in w for w in stats["warnings"])


def test_schema_sent_to_the_model_has_no_default_keys():
    assert '"default"' not in json.dumps(storyboard.SCHEMA)


def test_original_slides_are_off_by_default_and_never_shown(monkeypatch):
    monkeypatch.setattr(config, "ORIGINAL_SLIDES", False)
    visual = storyboard.schema_for_request()["properties"]["scenes"]["items"]["properties"]["visual"]
    assert "slide" not in visual["enum"]
    sb = storyboard.Storyboard.model_validate({"title": "T", "goal": "G", "scenes": [
        {"visual": "slide", "slide": "P1", "heading": "Joins", "items": ["a", "b"], "narration": "Talk.", "sources": []}]})
    scenes, _ = storyboard.repair(sb, {}, {"P1": {"image": "x.jpg", "title": "R", "start": 0}}, 1)
    assert scenes[1]["visual"] == "bullets" and "slide_image" not in scenes[1]


def test_new_slide_types_are_kept_when_complete_and_downgraded_when_not():
    sb = storyboard.Storyboard.model_validate({"title": "T", "goal": "G", "scenes": [
        {"visual": "comparison", "heading": "A vs B", "narration": "x.", "columns": [
            {"heading": "A", "items": ["one"]}, {"heading": "B", "items": ["two"]}]},
        {"visual": "comparison", "heading": "Only one", "narration": "x.", "columns": [
            {"heading": "A", "items": ["one"]}]},
        {"visual": "table", "heading": "Grid", "narration": "x.", "table_header": ["Need", "Join"],
         "table_rows": [["Matches", "Inner"], ["All rows", "Left", "extra cell"]]},
        {"visual": "table", "heading": "Too small", "narration": "x.", "table_header": ["Need"], "table_rows": [["a"]]},
        {"visual": "concept", "heading": "Join key", "items": ["A shared column"], "narration": "x."},
        {"visual": "concept", "heading": "", "items": ["no term"], "narration": "x."}]})
    scenes, _ = storyboard.repair(sb, {}, {}, 1)
    kinds = [s["visual"] for s in scenes[1:]]
    assert kinds == ["comparison", "bullets", "table", "bullets", "concept", "bullets"]
    assert scenes[3]["table_rows"][1] == ["All rows", "Left"]          # extra cell cut to the header
    assert scenes[2]["items"] == ["A: one"]                            # downgraded comparison keeps its content


def test_every_scene_type_renders_at_video_size(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SLIDE_MASK", "0.7,0.6,1,1")
    slide = tmp_path / "slide.jpg"
    Image.new("RGB", (1280, 720), "white").save(slide)
    long_text = "A very long on-screen point that keeps going " * 6
    scenes = [{"visual": "title", "heading": "Joining tables"},
              {"visual": "bullets", "heading": "Why", "items": ["One view", long_text]},
              {"visual": "bullets", "heading": "Only a heading"},
              {"visual": "steps", "heading": "How", "items": ["Pick key", "Clean key", "Join", "Check"]},
              {"visual": "steps", "heading": "Many", "items": [long_text] * 6},
              {"visual": "comparison", "heading": "A vs B", "columns": [
                  {"heading": "Inner join", "items": ["Only matches", long_text]},
                  {"heading": "Left join", "items": ["All rows of the first table"]},
                  {"heading": "Full join", "items": ["Everything"]}]},
              {"visual": "table", "heading": "Which join", "table_header": ["Question", "Join", "Result"],
               "table_rows": [["Which customers ordered?", "Inner", "Only matches"], [long_text, "Left", "x"]]},
              {"visual": "concept", "heading": "Join key", "items": ["A column both tables share", "Example: ID"]},
              {"visual": "slide"}]
    for i, sc in enumerate(scenes):
        out = render.render_scene(sc, tmp_path / f"s{i}.png", lesson_title="T", goal="G", page=f"{i}/9",
                                  slide_image=slide, caption="From: test")
        with Image.open(out) as im:
            assert im.size == (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)


def test_pronunciation_changes_speech_but_not_subtitles(project):
    config.PRONUNCIATION_FILE.write_text("term,say_as\nKPI,K P I\n# comment,x\n", encoding="utf-8")
    assert tts.clean_for_speech("The KPI and KPIs [S2]") == "The K P I and KPIs"
    assert tts.clean_text("The KPI [S2]") == "The KPI"


def test_read_aloud_and_unusual_words_checks():
    slide = "Combine two tables on a key. Key is a column both tables share."
    assert quality.read_aloud_ratio("Combine two tables on a key. Key is a column both tables share.", slide) == 1.0
    assert quality.read_aloud_ratio("We merge data using a common identifier.", slide) == 0.0
    assert quality.unusual_words("The KPI and VLOOKUP in Q3, with PowerBI. OK.", {"KPI"}) == ["PowerBI", "Q3", "VLOOKUP"]
