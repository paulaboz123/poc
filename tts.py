"""
tts.py - turns narration text into speech, locally (no audio or text leaves the laptop here).

Engines (TTS_ENGINE in .env):
- kokoro: Kokoro-82M, an open model (Apache 2.0) with natural English voices, run on the CPU
  with onnxruntime. Needs two files in models/ (~120 MB), fetched once from GitHub:
      uv add kokoro-onnx
      uv run tts.py --download
- system: the voice built into Windows (via pyttsx3). Robotic, but needs no download.
      uv add pyttsx3

Why local: Bedrock has no text-to-speech model, and Amazon Polly would need AWS access.

Usage:
  uv run tts.py --download                   # once
  uv run tts.py "Hello, this is a test."     # writes data/output/tts_test.wav and shows the speed
  uv run tts.py --voices                     # list the Kokoro voices (try a few, set TTS_VOICE)
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("tts")
logging.getLogger("phonemizer").setLevel(logging.ERROR)   # harmless "words count mismatch" noise

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_MODEL = "kokoro-v1.0.int8.onnx"   # 88 MB, fastest on a CPU; quality close to the full model
KOKORO_VOICES = "voices-v1.0.bin"        # 28 MB, all voices
PAUSE_AFTER_S = 0.45                     # silence after each scene, so scenes don't run together


class TTSError(RuntimeError):
    pass


# --- Model files ---------------------------------------------------------------------------
def kokoro_paths() -> tuple[Path, Path]:
    """Where the Kokoro model and voice files are expected (the models/ folder)."""
    return config.MODELS_DIR / KOKORO_MODEL, config.MODELS_DIR / KOKORO_VOICES


def download() -> None:
    """Fetch the Kokoro files from GitHub (honours HTTPS_PROXY and CA_BUNDLE)."""
    import llm   # reuse its HTTP settings (proxy, company certificate)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path in kokoro_paths():
        if path.exists() and path.stat().st_size > 1_000_000:
            log.info("%s already there", path.name)
            continue
        part = path.with_suffix(path.suffix + ".part")
        log.info("Downloading %s ...", path.name)
        with llm._http().stream("GET", KOKORO_RELEASE + path.name, follow_redirects=True,
                                timeout=600) as r:
            if r.status_code != 200:
                raise TTSError(f"Download failed (HTTP {r.status_code}). If GitHub is blocked, download "
                               f"{KOKORO_RELEASE + path.name} in the browser and put it in {config.MODELS_DIR}")
            with open(part, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        part.replace(path)
        log.info("Saved %s (%.0f MB)", path, path.stat().st_size / 1e6)


def check() -> tuple[bool, str]:
    """Is the configured engine usable? (for doctor.py)"""
    if config.TTS_ENGINE == "kokoro":
        if importlib.util.find_spec("kokoro_onnx") is None:
            return False, "kokoro-onnx not installed (uv add kokoro-onnx)"
        missing = [p.name for p in kokoro_paths() if not p.exists()]
        return (not missing, f"voice {config.TTS_VOICE}" if not missing else f"missing {missing}")
    if config.TTS_ENGINE == "system":
        if importlib.util.find_spec("pyttsx3") is None:
            return False, "pyttsx3 not installed (uv add pyttsx3)"
        return True, "Windows voice"
    return False, f"unknown TTS_ENGINE {config.TTS_ENGINE!r} (use kokoro or system)"


# --- Synthesis -------------------------------------------------------------------------------
_kokoro = None


def _get_kokoro():
    """Load Kokoro once and reuse it; checks the files and the configured voice."""
    global _kokoro
    if _kokoro is None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as e:
            raise TTSError("kokoro-onnx is not installed. Run: uv add kokoro-onnx") from e
        model, voices = kokoro_paths()
        if not model.exists() or not voices.exists():
            raise TTSError(f"Kokoro model files missing in {config.MODELS_DIR}. Run: uv run tts.py --download")
        _kokoro = Kokoro(str(model), str(voices))
        if config.TTS_VOICE not in _kokoro.get_voices():
            raise TTSError(f"Voice {config.TTS_VOICE!r} not found. See: uv run tts.py --voices")
    return _kokoro


def clean_text(text: str) -> str:
    """Narration as it should be shown (subtitles): no markdown, no source labels, single spaces."""
    text = re.sub(r"\[(?:S|P)\d+\]", "", text)
    text = re.sub(r"[*_#`>|]", "", text)
    return " ".join(text.split())


def load_pronunciations() -> dict[str, str]:
    """Terms and how to say them, from pronunciation.csv (columns: term,say_as).
    Example rows:  KPI,K P I    VLOOKUP,V lookup    SAP,S A P    Roche,Rosh"""
    path = config.PRONUNCIATION_FILE
    if not path.exists():
        return {}
    import csv
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if len(r) >= 2 and r[0].strip() and not r[0].startswith("#")]
    return {r[0].strip(): r[1].strip() for r in rows if r[0].strip().lower() != "term"}


def clean_for_speech(text: str) -> str:
    """Narration as it should be spoken: clean_text, common abbreviations written out, and the
    terms from pronunciation.csv replaced by how to say them (whole words, case-sensitive)."""
    text = clean_text(text)
    text = text.replace("&", " and ").replace("e.g.", "for example").replace("i.e.", "that is")
    for term, say_as in sorted(load_pronunciations().items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", say_as, text)
    return " ".join(text.split())


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """Save mono float samples (-1..1) as a 16-bit .wav file."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a 16-bit .wav as mono float samples (-1..1). Returns (samples, sample rate)."""
    with wave.open(str(path), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise TTSError(f"Unexpected sample width {width} in {path.name}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def _system_tts(text: str) -> tuple[np.ndarray, int]:
    """Speak with the voice built into Windows (pyttsx3). Returns (samples, sample rate)."""
    try:
        import pyttsx3
    except ImportError as e:
        raise TTSError("pyttsx3 is not installed. Run: uv add pyttsx3") from e
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "speech.wav"
        engine = pyttsx3.init()
        engine.setProperty("rate", int(170 * config.TTS_SPEED))
        engine.save_to_file(text, str(out))
        engine.runAndWait()
        engine.stop()
        if not out.exists():
            raise TTSError("The system voice produced no audio")
        return _read_wav(out)


def synthesize(text: str) -> tuple[np.ndarray, int]:
    """Mono float samples and sample rate for a text."""
    text = clean_for_speech(text)
    if not text:
        raise TTSError("Nothing to say: the narration is empty")
    if config.TTS_ENGINE == "kokoro":
        samples, rate = _get_kokoro().create(text, voice=config.TTS_VOICE, speed=config.TTS_SPEED,
                                             lang="en-us")
        return np.asarray(samples, dtype=np.float32), rate
    if config.TTS_ENGINE == "system":
        return _system_tts(text)
    raise TTSError(f"Unknown TTS_ENGINE={config.TTS_ENGINE!r}, use kokoro or system")


def speak_to_wav(text: str, out: Path, pause_after_s: float = PAUSE_AFTER_S) -> float:
    """Speak a text into a .wav file (with a short pause at the end). Returns its length in s."""
    samples, rate = synthesize(text)
    lead = np.zeros(int(0.15 * rate), dtype=np.float32)
    tail = np.zeros(int(pause_after_s * rate), dtype=np.float32)
    audio = np.concatenate([lead, samples, tail])
    _write_wav(out, audio, rate)
    return len(audio) / rate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="text to speak as a test")
    ap.add_argument("--download", action="store_true", help="download the Kokoro model files")
    ap.add_argument("--voices", action="store_true", help="list Kokoro voices")
    args = ap.parse_args()
    config.setup_logging()
    try:
        if args.download:
            download()
        if args.voices:
            names = sorted(_get_kokoro().get_voices())
            print("English voices (a = American, b = British; f = female, m = male):")
            print("  " + ", ".join(v for v in names if v[:1] in "ab"))
        if args.text:
            t0 = time.perf_counter()
            out = config.OUTPUT_DIR / "tts_test.wav"
            secs = speak_to_wav(args.text, out)
            took = time.perf_counter() - t0
            log.info("Wrote %s: %.1fs of speech in %.1fs (%s, voice %s)", out, secs, took,
                     config.TTS_ENGINE, config.TTS_VOICE)
    except Exception as e:   # noqa: BLE001 - show a readable message, not a traceback
        log.error("%s", e)
        return 1
    if not (args.download or args.voices or args.text):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
