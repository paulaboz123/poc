# Video lesson PoC

Turns training recordings (video + transcript) into **short narrated video lessons**, an
**audio lesson** and a **text script**. One code base, two ways to run the models, chosen in `.env`:

| Setting | Models | Needs | Quality / speed |
|---|---|---|---|
| **Local** (`.env.local.example`) | Llama 3.1 (Meta), Gemma 3 (Google), nomic-embed-text, all in Ollama | nothing | OK / slow (minutes per lesson) |
| **Gateway** (`.env.bifrost.example`) | Claude Sonnet 5 on Amazon Bedrock via Bifrost; search stays local | URL + key + model id from the platform team | good / fast (seconds), ~$0.05 per video |

No models of Chinese origin are used, locally or on the gateway. Voice (Kokoro) and video
(ffmpeg) always run on the laptop.

```
data/input  (mp4 + vtt/docx + description.xlsx)
   │
   │  pipeline.py      1 inventory: pair videos, transcripts and Excel descriptions
   │                   2 transcript → ~1-minute segments
   │                   3 slides: find them in the video, read their text (OCR)
   │                   4 (optional --vision) describe what each slide shows
   │                   5 search index
   ▼
make_video.py "topic"  or  --recording 13
   1 material     transcript parts + the slides shown at that moment
   2 storyboard   the model plans scenes: what the voice says + what the screen shows
   3 voice        Kokoro speaks each scene
   4 pictures     original slide / key points / numbered steps
   5 video        lesson_video.mp4 (+ .srt), lesson_audio.mp3, script.md
```

---

## Part A: run everything locally (start here)

### A1. One-time setup

1. **Copy the files** from the zip into your project folder (overwrite the `.py` files and
   `skills/`). Keep your `data/` folder.
2. **Install Python packages** (one command at a time):
   ```
   uv add httpx
   uv add python-dotenv
   uv add python-docx
   uv add numpy
   uv add pillow
   uv add pydantic
   uv add openpyxl
   uv add kokoro-onnx
   uv add --dev pytest
   ```
   Already installed ones finish in a second.
3. **Local models** (Ollama app running):
   ```
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
   Optional, only for slide descriptions: `ollama pull gemma3:4b`.
4. **Settings:** copy `.env.local.example` and rename the copy to `.env`.
5. **Voice** (two files from GitHub, ~120 MB, once):
   ```
   uv run tts.py --download
   uv run tts.py "Joining data means combining two tables on a shared key."
   ```
   Listen to `data/output/tts_test.wav`. Other voices: `uv run tts.py --voices`, then set
   `TTS_VOICE` in `.env`. If GitHub is blocked, download `kokoro-v1.0.int8.onnx` and
   `voices-v1.0.bin` in the browser from
   https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0 into `models/`.
6. **Check:**
   ```
   uv run doctor.py
   uv run pytest
   ```
   `doctor.py`: every line `[OK]` (a `[WARN]` for the optional vision model is fine).
   `pytest`: runs 25 automatic tests in about 30 s, offline and free (stand-in models, a made-up
   recording). Expect `25 passed`. Run it again after every code change.

### A2. Prepare the recordings

7. Put the files in `data/input/` (sub-folders are fine): the slide videos (`.mp4`), their
   transcripts (`.vtt`/`.docx`, same name as the video), and `description.xlsx` (columns `id`, `name`, `description`).
8. ```
   uv run inventory.py
   ```
   A table appears. Every video should have an id, a transcript and `OK`. Open
   `data/manifest.csv` and set `include` to `yes` for the videos to use (screen demos without slides: `no`).
9. **Slide area**, for one video (the `video_id` is in the manifest, e.g. `13-introduction-to-joining-data-d25c`):
   ```
   uv run pipeline.py --mode easy
   uv run slides.py --video <video_id> --preview
   ```
   Open `data/processed/<video_id>/crop_preview.jpg`. The red box is what is read as the slide.
   - presenter or Teams panel **next to** the slide → set `SLIDE_CROP` in `.env`, then
     `uv run pipeline.py --mode easy --force`
   - presenter camera **on top of** the slide → set `SLIDE_MASK` (blue box); it is hidden in the video
   - look at `data/processed/<video_id>/slides/`: each slide should appear once, complete
10. **Search check:**
    ```
    uv run index.py search "how to join two tables" -k 5
    ```
    The right recording should be on top.

### A3. Make lessons and videos

11. **Plan a video first** (cheap check), then make it:
    ```
    uv run make_video.py --recording 13 --minutes 2 --plan-only
    ```
    Read `data/output/<folder>/script.md`. If it's fine:
    ```
    uv run make_video.py --rerender "data/output/<folder>"
    ```
    Watch `lesson_video.mp4`, listen to `lesson_audio.mp3`.
12. **Topic across all recordings, in one go:**
    ```
    uv run make_video.py "how to join two tables" --minutes 2
    ```
13. **Text lesson:**
    ```
    uv run lesson.py "how to join two tables" --minutes 5 --level beginner
    ```

To fix a sentence or on-screen word in a video: edit `storyboard.json` in the output folder
(`narration`, `heading`, `items`) and run `--rerender`. No model call.

---

## Part B: switch to the gateway (Claude on Bedrock)

You need the 3 values from the platform team's e-mail: the **address** (`ANTHROPIC_BASE_URL`),
the **virtual key**, and the **Sonnet 5 model id (EU)**. Search keeps running locally
(Claude cannot make embeddings), so Ollama with `nomic-embed-text` stays installed.

1. Copy `.env.bifrost.example` and rename the copy to `.env` (the old `.env` is replaced;
   copy your `SLIDE_CROP` / `SLIDE_MASK` values over first).
2. Fill in 3 lines, values exactly as in the e-mail:
   ```
   BIFROST_BASE_URL=<the address; the /anthropic ending can stay>
   BIFROST_API_KEY=<the virtual key>
   CHAT_MODEL=<the Sonnet 5 id, e.g. bedrock/eu.anthropic.claude-sonnet-...>
   ```
3. Check:
   ```
   uv run doctor.py --models
   ```
   Look for `[OK] chat model answers: 'OK'`. A model list may or may not appear; either is fine.
   If you get HTTP 401: set `BIFROST_AUTH_HEADER=x-bf-vk` and try again.
4. Prepare again with Claude reading the slides (segments and slides are kept):
   ```
   uv run pipeline.py --mode easy --vision
   ```
5. Same commands as in A3:
   ```
   uv run make_video.py --recording 13 --minutes 2 --plan-only
   uv run make_video.py --rerender "data/output/<folder>"
   ```
   Compare `script.md` with the local one: that comparison is useful for the demo.

**Other models:** for one run add `--model <id>` (e.g. the Opus 5 id from the e-mail); to switch
for good, change `CHAT_MODEL`. Use EU ids only (the platform team asked for that).

### Models and cost

| Job | Model | Cost (estimate) |
|---|---|---|
| Storyboard, text lesson, reading slides | Claude Sonnet 5 (EU) | ~$0.03-0.06 per 2-min video; ~$0.006 per slide, once |
| Search | nomic-embed-text, local | free |
| Comparison now and then | Claude Opus 5 | ~2.5x Sonnet |

### Spending lock

| Setting | Default | What happens |
|---|---|---|
| `DAILY_BUDGET_USD` | 20 | no more gateway calls today once reached |
| `RUN_BUDGET_USD` | 5 | one command stops once it has spent this much (protects against loops) |
| hard maximum | 100 | fixed in the code: a higher `DAILY_BUDGET_USD` is cut down to 100 |

Costs are estimated from token counts with list prices (+10% for EU ids, and the higher price
when a version is unknown), logged per call in `cost_log.csv`. `uv run doctor.py` shows today's
total. Only this pipeline is counted: Claude Code on the same key counts toward the platform's
own limit too.

---

## Part C: work in batches, check quality, report

**Many lessons at once.** Copy `requests.example.csv` to `requests.csv` and edit it in Excel or
VS Code: one row per lesson (`type` video or lesson, `topic` or `recording`, `minutes`, `level`,
optional `model`, `run` yes/no). New lessons = new rows.
```
uv run batch.py --plan-only      # storyboards only: fast and cheap; read the scripts first
uv run batch.py                  # everything: video, audio, script, checks
uv run batch.py --review         # plus a model review of every video (~$0.01 each)
uv run batch.py --only 2,5       # just some rows
```
Every row runs on its own: a failed row is recorded and the batch goes on; only the spending
lock stops it. At the end: a summary table, `data/output/batch_<time>.csv` and the report.

**Quality checks** (also part of `batch.py`): `uv run quality.py --all [--review]`, or
`make_video.py ... --check` for one video. Status per video:

| Status | Meaning |
|---|---|
| OK | nothing found |
| CHECK | worth a look: length off, reads a slide aloud, text too long on screen, words to listen to, review score 3 or lower |
| FAIL | broken (missing or out-of-sync video) or the review found invented content |
| PLAN | storyboard only, not rendered yet |

**Report:** `uv run report.py` writes `data/output/report.html`: double-click it. It shows every
video with status, problems, review scores, a thumbnail of each scene (hover = narration), links to
video/audio/script, text lessons, and the estimated spend per step.

**Pronunciation:** if the voice says a term wrongly, add a row to `pronunciation.csv`
(`KPI,K P I`), test with `uv run tts.py "The KPI ..."`, then `--rerender`. Subtitles keep the
original spelling. The quality check lists unusual words to listen to.

**NotebookLM route:** `uv run export_notebooklm.py [--combined]` writes the prepared recordings
as `.md` files in `data/output/notebooklm/` (transcript with timestamps, slide texts, no speaker
names). Upload them to NotebookLM as sources (it does not take MP4). Upload only what you are
allowed to: NotebookLM is a Google service outside the platform.

---

## Commands at a glance

| Command | What it does |
|---|---|
| `uv run doctor.py [--models]` | checks the setup (and lists available models) |
| `uv run inventory.py` | finds recordings, pairs transcripts and Excel rows → `data/manifest.csv` |
| `uv run pipeline.py --mode easy [--vision] [--force]` | prepares all included recordings |
| `uv run slides.py --video <id> --preview` | shows the slide area (crop) and the hidden area (mask) |
| `uv run index.py search "..."` | tests search |
| `uv run make_video.py "topic" --minutes 2` | video from all recordings |
| `uv run make_video.py --recording 13 --minutes 2` | video from one recording (Excel id) |
| `uv run make_video.py ... --plan-only` / `--rerender <folder>` | plan only / voice + video from an (edited) plan |
| `uv run lesson.py "topic" --minutes 5 --level beginner` | text lesson |
| `uv run batch.py [--plan-only] [--review]` | all rows of `requests.csv`, checked, with report |
| `uv run quality.py --all [--review]` | checks every video (status OK / CHECK / FAIL / PLAN) |
| `uv run report.py` | `data/output/report.html` with everything |
| `uv run export_notebooklm.py` | `.md` sources for NotebookLM |
| `uv run pytest` | automatic tests (offline, free, ~30 s) |
| `uv run tts.py "text"` | voice test |
| `uv run llm.py` | model smoke test |

Levels: `beginner`, `intermediate`, `expert`.

## Output of a video

Folder `data/output/<time>-<topic>-video-<n>m/`:

| File | What |
|---|---|
| `lesson_video.mp4` | the video lesson (subtitles inside as a track) |
| `lesson_video.srt` | subtitles |
| `lesson_audio.mp3` | audio lesson |
| `script.md` | narration + on-screen text per scene, with sources (recording + time) |
| `storyboard.json` | the same as data; edit it and `--rerender` |
| `info.json` | length, timings, estimated cost, warnings |

## When something fails

| Message | Fix |
|---|---|
| `Is Ollama running?` | start the Ollama app |
| `ollama pull ...` in doctor | the model isn't downloaded yet: run that command |
| `Prompt is ~N tokens ... exceeds NUM_CTX` | lower `MAX_MATERIAL_WORDS` or raise `NUM_CTX` in `.env` |
| `HTTP 401/403 ... rejected` | wrong key or model not allowed for it: check `.env`, ask the platform team |
| `HTTP 404 ... model id unknown` | copy the exact id from `uv run doctor.py --models` |
| `CERTIFICATE_VERIFY_FAILED` | company TLS inspection: set `CA_BUNDLE` to the company `.pem` |
| `Can this laptop reach the gateway` | VPN/proxy, or the URL works only inside their cluster |
| `Index was built with ...` | embedding model changed: `uv run index.py build` |
| `Recordings don't seem to cover` | other wording, or lower `MIN_SCORE` |
| `Spending lock: ...` | daily or per-command limit reached; raise it in `.env` only if expected (max 100/day) |
| `CHAT_MODEL is empty` | paste the Sonnet id from the e-mail |
| `Both gateway routes failed` | check the address and the model id; send the message (without the key) |
| presenter visible on slides in the video | `SLIDE_MASK`, then `--rerender` |
| very slow storyboard locally | `CHAT_MODEL=llama3.2:3b`, or shorter `--minutes` |

## Files

| File | Role |
|---|---|
| `config.py` | all settings, read from `.env` |
| `llm.py` | every model call: gateway or Ollama, retries, cost log, budget stop |
| `inventory.py`, `ingest.py`, `slides.py`, `describe_slides.py`, `index.py`, `pipeline.py`, `transcribe.py` | preparing recordings |
| `lesson.py` | text lesson |
| `storyboard.py`, `tts.py`, `render.py`, `make_video.py` | narrated video |
| `quality.py`, `report.py`, `batch.py` | checks, report, many lessons from `requests.csv` |
| `export_notebooklm.py` | NotebookLM source files |
| `pronunciation.csv` | how the voice says special terms |
| `tests/` | automatic tests with a stand-in gateway and voice |
| `store.py`, `media.py` | files on disk; ffmpeg / Tesseract |
| `skills/compose-video/`, `skills/compose-lesson/` | the prompts (SKILL.md, the platform's format) |
| `evaluate.py`, `eval/` | scoring lessons against a golden set |
| `doctor.py` | checks the setup |
