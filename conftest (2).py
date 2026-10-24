"""
Shared test setup. Every test runs in its own temporary project folder with a fake gateway
and a fake voice, so the tests are offline, free, fast, and never touch your real data.
"""
from __future__ import annotations

import faulthandler
import shutil
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the project folder

import config  # noqa: E402
import llm  # noqa: E402
import tts  # noqa: E402
from fake_gateway import Gateway  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which(config.FFMPEG) is None, reason="ffmpeg not installed")


@pytest.fixture(autouse=True)
def watchdog():
    """If one test runs longer than 3 minutes, print where every thread is stuck and stop,
    instead of hanging silently (Ctrl+C can't always interrupt on Windows)."""
    faulthandler.dump_traceback_later(180, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Point every data folder to an empty temporary project and reset model-call state."""
    # Company laptops often set a proxy for all web traffic. The stand-in gateway runs on this
    # computer, so requests to it must not go through the proxy (they would hang until timeout).
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    data = tmp_path / "data"
    for name, path in {"DATA_DIR": data, "INPUT_DIR": data / "input", "PROCESSED_DIR": data / "processed",
                       "INDEX_DIR": data / "index", "OUTPUT_DIR": data / "output",
                       "MANIFEST": data / "manifest.csv", "COST_LOG": tmp_path / "cost_log.csv",
                       "PRONUNCIATION_FILE": tmp_path / "pronunciation.csv"}.items():
        monkeypatch.setattr(config, name, path)
    (data / "input").mkdir(parents=True)
    # a connection that ignores every proxy setting (also the Windows system proxy)
    monkeypatch.setattr(llm, "_client", httpx.Client(trust_env=False, timeout=httpx.Timeout(10, connect=5)))
    for name, value in {"_spent_today": None, "_spent_session": 0.0, "_route": None,
                        "_structured_ok": True, "_embed_one_by_one": False}.items():
        monkeypatch.setattr(llm, name, value)
    monkeypatch.setattr(config, "DAILY_BUDGET_USD", 20.0)
    monkeypatch.setattr(config, "RUN_BUDGET_USD", 5.0)
    return tmp_path


@pytest.fixture
def gateway(project, monkeypatch):
    """A fake Bifrost gateway; all model jobs go to it."""
    gw = Gateway()
    settings = {"BIFROST_BASE_URL": gw.url, "BIFROST_API_KEY": "test-key", "BIFROST_AUTH_HEADER": "auto",
                "GATEWAY_API": "auto", "CHAT_BACKEND": "bifrost", "VISION_BACKEND": "bifrost",
                "EMBED_BACKEND": "bifrost", "CHAT_MODEL": "bedrock/eu.anthropic.claude-sonnet-5-test",
                "VISION_MODEL": "bedrock/eu.anthropic.claude-sonnet-5-test", "REVIEW_MODEL": "",
                "EMBED_MODEL": "bedrock/amazon.titan-embed-text-v2:0", "MIN_SCORE": 0.0,
                "MAX_MATERIAL_WORDS": 9000, "LLM_TIMEOUT_S": 10.0, "CA_BUNDLE": ""}
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)
    yield gw
    gw.stop()


@pytest.fixture
def fake_voice(monkeypatch):
    """Replace the voice with a quiet tone: 0.3 s per word, so no model files are needed."""
    def synthesize(text: str):
        rate = 24000
        n = int(len(tts.clean_for_speech(text).split()) * 0.3 * rate)
        return (0.1 * np.sin(np.arange(n) / rate * 2 * np.pi * 220)).astype(np.float32), rate
    monkeypatch.setattr(tts, "synthesize", synthesize)
