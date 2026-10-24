"""
config.py - one place for every path and setting.

Every other script does `import config` and reads from here. Settings come from .env, so
you change things in one place. Start from a template:
  .env.local.example    everything on this laptop (Ollama), no key needed
  .env.bifrost.example  Claude + Titan on Bedrock via the Bifrost gateway (needs URL + key)
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    """Read a setting from .env (or the environment), falling back to a default."""
    value = os.getenv(name, "").strip()
    return value or default


def _find_tesseract() -> str:
    """TESSERACT from .env, else whatever is on PATH, else the default Windows folder."""
    configured = _get("TESSERACT")
    if configured:
        return configured
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    # usual Windows install folders: for all users, 32-bit, or only for this user (winget/no admin)
    local = os.getenv("LOCALAPPDATA", "")
    for folder in (r"C:\Program Files\Tesseract-OCR", r"C:\Program Files (x86)\Tesseract-OCR",
                   local and os.path.join(local, "Programs", "Tesseract-OCR"),
                   local and os.path.join(local, "Tesseract-OCR")):
        if folder and Path(folder, "tesseract.exe").exists():
            return str(Path(folder, "tesseract.exe"))
    return "tesseract"


# --- Folders ------------------------------------------------------------------
DATA_DIR = ROOT / "data"
INPUT_DIR = DATA_DIR / "input"          # original recordings + transcripts (never modified)
PROCESSED_DIR = DATA_DIR / "processed"  # one sub-folder per recording: segments, slides
INDEX_DIR = DATA_DIR / "index"          # search index (embeddings)
OUTPUT_DIR = DATA_DIR / "output"        # generated lessons and videos
MANIFEST = DATA_DIR / "manifest.csv"    # which recording goes with which transcript
MODELS_DIR = ROOT / "models"            # local model files (the Kokoro voice)
SKILLS_DIR = ROOT / "skills"            # prompts, in SKILL.md format
EVAL_DIR = ROOT / "eval"                # golden set + evaluation results
COST_LOG = ROOT / "cost_log.csv"        # one row per model call
# Excel with columns id | name | description, one row per recording (see inventory.py).
# Empty = the first description*.xlsx found anywhere in data/input.
DESCRIPTIONS_FILE = _get("DESCRIPTIONS_FILE")

# --- Model access -----------------------------------------------------------------
# Every model job can run on the company gateway or on this laptop:
#   bifrost = Bifrost gateway -> Amazon Bedrock (Claude). Needs URL + key below.
#   ollama  = local models via Ollama: free and offline, but slow and weaker.
# Paste the values from the platform team's e-mail as they are: the Claude Code address
# ".../anthropic" works too, the ending is removed here.
def _gateway_base(url: str) -> str:
    """'https://x/anthropic/' -> 'https://x' (also strips /openai and /v1 endings)."""
    url = url.strip().rstrip("/")
    for ending in ("/v1", "/anthropic", "/openai", "/v1"):
        url = url.removesuffix(ending).rstrip("/")
    return url


BIFROST_BASE_URL = _gateway_base(_get("BIFROST_BASE_URL", _get("ANTHROPIC_BASE_URL")))
BIFROST_API_KEY = _get("BIFROST_API_KEY", _get("ANTHROPIC_API_KEY"))
# How the key is sent: auto (both Authorization: Bearer and x-bf-vk, the default),
# authorization, x-bf-vk or x-api-key. Change only if the platform team says so.
BIFROST_AUTH_HEADER = _get("BIFROST_AUTH_HEADER", "auto").lower()
# Which gateway API to call: auto (OpenAI-style /v1, falls back to the Anthropic-style
# /anthropic route that Claude Code uses), openai, or anthropic.
GATEWAY_API = _get("GATEWAY_API", "auto").lower()
# Corporate networks sometimes inspect TLS with their own certificate. If you get
# "CERTIFICATE_VERIFY_FAILED", put the path of the company CA bundle (.pem) here.
CA_BUNDLE = _get("CA_BUNDLE")
OLLAMA_URL = _get("OLLAMA_URL", "http://localhost:11434").rstrip("/").removesuffix("/v1")

BACKENDS = ("bifrost", "ollama")
CHAT_BACKEND = _get("CHAT_BACKEND", "bifrost" if BIFROST_BASE_URL else "ollama").lower()
VISION_BACKEND = _get("VISION_BACKEND", CHAT_BACKEND).lower()
# Search runs locally by default: Claude models cannot make embeddings, and the local
# nomic-embed-text is free and good enough. Bedrock embeddings (Titan) only if enabled for you.
EMBED_BACKEND = _get("EMBED_BACKEND", "ollama").lower()

# Model names exactly as the backend expects them. For Bifrost: copy the ids from the
# platform team's e-mail (e.g. bedrock/eu.anthropic.claude-sonnet-...). No guessed defaults.
# Local defaults are from Meta (Llama) and Google (Gemma); models of Chinese origin are not
# allowed in this project, even offline.
CHAT_MODEL = _get("CHAT_MODEL", _get("ANTHROPIC_DEFAULT_SONNET_MODEL",
                  "" if CHAT_BACKEND == "bifrost" else "llama3.1:8b"))
if VISION_BACKEND == "bifrost":
    VISION_MODEL = _get("VISION_MODEL", CHAT_MODEL if CHAT_BACKEND == "bifrost" else "")
else:
    VISION_MODEL = _get("VISION_MODEL", "gemma3:4b")
EMBED_MODEL = _get("EMBED_MODEL", "bedrock/amazon.titan-embed-text-v2:0" if EMBED_BACKEND == "bifrost"
                   else "nomic-embed-text")
MODEL_ID = CHAT_MODEL   # old name, still used in a few log lines

NUM_CTX = int(_get("NUM_CTX", "16384"))               # Ollama only: context window we request
MAX_OUTPUT_TOKENS = int(_get("MAX_OUTPUT_TOKENS", "6000"))
LLM_TIMEOUT_S = float(_get("LLM_TIMEOUT_S", "1800" if CHAT_BACKEND == "ollama" else "300"))
# How much transcript (in words) a lesson or video may read. Claude reads a lot quickly
# and cheaply; a small local model needs short prompts.
MAX_MATERIAL_WORDS = int(_get("MAX_MATERIAL_WORDS", "9000" if CHAT_BACKEND == "bifrost" else "3000"))

# Search scores below this mean "the recordings don't cover this topic". The right value
# depends on the embedding model: check real scores with  uv run index.py search "..."
MIN_SCORE = float(_get("MIN_SCORE", "0.45" if "nomic" in EMBED_MODEL else "0.2"))

# --- Costs and spending lock ----------------------------------------------------------
# Estimated USD per 1M tokens (input, output), matched by a part of the model name, first
# match wins. Claude list prices; unknown versions get the higher price, so the lock errs
# on the safe side. Regional ids (eu./us. ...) get +10%. Local models cost 0.
PRICES: list[tuple[str, float, float]] = [
    ("sonnet-5", 2.0, 10.0), ("sonnet", 3.0, 15.0),
    ("haiku-5", 0.10, 0.50), ("haiku", 1.0, 5.0),
    ("opus-5-5", 4.0, 20.0), ("opus-4-1", 15.0, 75.0), ("opus-4-0", 15.0, 75.0), ("opus", 5.0, 25.0),
    ("titan-embed", 0.02, 0.0), ("cohere.embed", 0.10, 0.0),
]
REGIONAL_PREMIUM = 1.1
# Optional override for the chat model, if the platform team gives you exact prices
PRICE_IN_PER_MTOK = float(_get("PRICE_IN_PER_MTOK", "0"))
PRICE_OUT_PER_MTOK = float(_get("PRICE_OUT_PER_MTOK", "0"))

# Spending lock for gateway calls (estimated from token counts, logged in cost_log.csv):
# - DAILY_BUDGET_USD: no more calls today once this is reached (default 20)
# - RUN_BUDGET_USD:   one command stops once it has spent this much (default 5)
# - HARD_DAILY_LIMIT_USD: 100, fixed in code. .env values above it are cut down to it.
# Only this pipeline is counted here; Claude Code on the same key counts toward the
# platform's own limit as well.
HARD_DAILY_LIMIT_USD = 100.0


def clamp_budget(value: float) -> float:
    """A budget setting forced into 0..HARD_DAILY_LIMIT_USD (a typo can't remove the lock)."""
    return max(0.0, min(value, HARD_DAILY_LIMIT_USD))


DAILY_BUDGET_USD = clamp_budget(float(_get("DAILY_BUDGET_USD", "20")))
RUN_BUDGET_USD = min(clamp_budget(float(_get("RUN_BUDGET_USD", "5"))), DAILY_BUDGET_USD)
# Model for the automatic review of videos (quality.py --review); default: the chat model
REVIEW_MODEL = _get("REVIEW_MODEL", CHAT_MODEL)

# --- External tools -----------------------------------------------------------------
FFMPEG = _get("FFMPEG", "ffmpeg")
FFPROBE = _get("FFPROBE", "ffprobe")
TESSERACT = _find_tesseract()
OCR_LANG = _get("OCR_LANG", "eng")

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
AUDIO_EXT = {".m4a", ".mp3", ".wav", ".aac"}
MEDIA_EXT = VIDEO_EXT | AUDIO_EXT
TRANSCRIPT_EXT = {".vtt", ".docx"}

# --- Pipeline mode ------------------------------------------------------------------
# easy  = videos that already have a transcript (start here)
# audio = + audio files that already have a transcript
# all   = + every recording without a transcript (transcribed locally with Whisper)
PIPELINE_MODE = _get("PIPELINE_MODE", "easy")
MODES = ("easy", "audio", "all")

# Local speech-to-text, only used in mode "all": tiny / base / small / medium
WHISPER_MODEL = _get("WHISPER_MODEL", "small")

# Part of the frame that holds the slides, as fractions "left,top,right,bottom".
# Empty = whole frame. Teams recording with people on the right: e.g. 0,0,0.78,1
# A row's own "crop" column in the manifest overrides this.
SLIDE_CROP = _get("SLIDE_CROP")
# Part of the slide (after the crop) to paint over in the video lesson, same format.
# For a presenter's camera picture on top of the slide, e.g. bottom right: 0.72,0.62,1,1
# Check it with: uv run slides.py --video <video_id> --preview
SLIDE_MASK = _get("SLIDE_MASK")

# --- Narrated video -------------------------------------------------------------------
# kokoro = natural local voice (needs: uv run tts.py --download, once)
# system = the voice built into Windows (robotic, but needs nothing extra)
# no (default) = the video shows only NEW slides designed from the content; yes = the model may
# also show original slide images (screenshots from the recording)
ORIGINAL_SLIDES = _get("ORIGINAL_SLIDES", "no").lower() in ("yes", "true", "1")
TTS_ENGINE = _get("TTS_ENGINE", "kokoro").lower()
# How to say special terms (acronyms, product names): two columns term,say_as
PRONUNCIATION_FILE = ROOT / "pronunciation.csv"
TTS_VOICE = _get("TTS_VOICE", "af_heart")       # Kokoro voices: af_heart, af_sarah, am_michael, bf_emma ...
TTS_SPEED = float(_get("TTS_SPEED", "1.0"))
VIDEO_WIDTH = int(_get("VIDEO_WIDTH", "1280"))  # 1280x720 renders fast; 1920 for full HD
VIDEO_HEIGHT = VIDEO_WIDTH * 9 // 16


def setup_logging(verbose: bool = False) -> None:
    """Same log format in every script: time, level, module, message."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide one log line per HTTP request
