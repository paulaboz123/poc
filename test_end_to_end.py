"""The whole flow on a tiny made-up recording: transcript -> index -> video, text lesson,
quality check with review, batch, report and NotebookLM export."""
from __future__ import annotations

from conftest import needs_ffmpeg

import ask
import batch
import config
import export_notebooklm
import index
import ingest
import lesson
import llm
import make_video
import media
import quality
import report
import store

VTT = "WEBVTT\n\n" + "\n\n".join(
    f"00:{i:02d}:00.000 --> 00:{i:02d}:50.000\n<v Anita Rao>{text}</v>" for i, text in enumerate([
        "Joining data means combining two tables on a shared key column such as the customer ID.",
        "Before you join, trim spaces in the key and make sure both sides use the same data type.",
        "An inner join keeps only matching rows, a left join keeps every row of the first table.",
        "After the join, count the rows before and after and look at the keys that did not match.",
    ]))


def _prepare_one_recording() -> str:
    (config.INPUT_DIR / "13. Joining data.vtt").write_text(VTT, encoding="utf-8")
    row = {"video_id": "13-joining-data-abcd", "include": "yes", "catalog_id": "13", "title": "Joining data",
           "description": "How to join two tables.", "media_type": "none", "video_file": "",
           "transcript_file": "13. Joining data.vtt"}
    store.write_manifest([row])
    assert ingest.ingest_row(row, force=False) == "done"
    assert index.build() == 0
    return row["video_id"]


@needs_ffmpeg
def test_video_from_a_topic_end_to_end(gateway, fake_voice):
    _prepare_one_recording()
    out_dir, info = make_video.make("how to join two tables", minutes=1)
    for name in ("lesson_video.mp4", "lesson_video.srt", "storyboard.json", "script.md", "info.json"):
        assert (out_dir / name).exists(), name
    assert (out_dir / "lesson_audio.mp3").exists() or (out_dir / "lesson_audio.m4a").exists()
    tracks = media.stream_durations(out_dir / "lesson_video.mp4")
    assert abs(tracks["video"] - tracks["audio"]) < 0.5 and tracks["video"] > 10
    assert info["model_cost_usd_est"] > 0

    result = quality.check(out_dir, review=True)
    assert result["status"] in ("OK", "CHECK") and result["review"]["faithful"] == 4
    assert quality.check(out_dir)["review"]["faithful"] == 4      # review kept without a new call


@needs_ffmpeg
def test_video_from_one_recording_and_plan_then_rerender(gateway, fake_voice):
    _prepare_one_recording()
    out_dir, info = make_video.make(recording="13", minutes=1, plan_only=True)
    assert info["plan_only"] and not (out_dir / "lesson_video.mp4").exists()
    calls_before = len(gateway.state["calls"])
    make_video.make(rerender=out_dir)
    assert (out_dir / "lesson_video.mp4").exists()
    assert len(gateway.state["calls"]) == calls_before            # re-render needs no model call


def test_text_lesson_and_not_covered_topic(gateway, monkeypatch):
    _prepare_one_recording()
    record = lesson.make_lesson("how to join two tables", minutes=2)
    assert record["stats"]["invalid_citations"] == [] and (config.OUTPUT_DIR / record["lesson_id"] / "lesson.md").exists()
    monkeypatch.setattr(config, "MIN_SCORE", 0.99)
    try:
        lesson.make_lesson("quantum chemistry", minutes=2)
        raise AssertionError("expected NotEnoughMaterial")
    except lesson.NotEnoughMaterial:
        pass


@needs_ffmpeg
def test_batch_report_and_notebooklm_export(gateway, fake_voice):
    vid = _prepare_one_recording()
    requests = config.ROOT / "tests" / "_requests.csv"
    requests.write_text("id,type,topic,recording,minutes,level,model,run\n"
                        "1,video,,13,1,beginner,,yes\n2,lesson,how to join two tables,,2,beginner,,yes\n"
                        "3,video,joins,,1,wizard,,yes\n4,video,joins,,1,beginner,,no\n", encoding="utf-8")
    try:
        rows = [r for r in batch.read_requests(requests) if r["run"] == "yes"]
    finally:
        requests.unlink()
    results = [batch.run_one(r, plan_only=False, review=False) for r in rows]
    assert [r["status"] in ("OK", "CHECK") for r in results] == [True, True, False]   # bad level -> FAIL, batch goes on
    page = report.build()
    assert "Joining tables" in page.read_text(encoding="utf-8")
    md = export_notebooklm.recording_markdown(vid)
    assert md.startswith("# Joining data") and "Anita Rao" not in md and "00:00 - " in md
    assert llm.spent_today() > 0


def test_recordings_found_by_title_file_name_or_id_and_listed_when_not(gateway):
    vid = _prepare_one_recording()
    assert store.resolve_recordings("13") == [vid]
    assert store.resolve_recordings("JOINING") == [vid]
    try:
        store.resolve_recordings("quantum")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "Available recordings" in str(e) and "Joining data" in str(e)


@needs_ffmpeg
def test_request_in_plain_words_is_understood_and_planned(gateway, fake_voice):
    vid = _prepare_one_recording()
    s = ask.understand("Hej, chce 1 minute o laczeniu danych, jako prezentacja z tabela")
    assert s["recordings"] == [vid] and s["level"] == "beginner" and s["format"] == "video"   # bad values fixed
    out_dir, info = make_video.make(s["topic"], ",".join(s["recordings"]), s["minutes"], s["level"],
                                    plan_only=True, wishes=s["wishes"])
    assert info["plan_only"] and info["request"]["wishes"] == "use a comparison table"
