"""
A stand-in for the Bifrost gateway, used only by the tests: same routes and answer formats,
canned answers, no real model and no cost. Behaviour can be switched per test via `state`.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = "test-key"
MODELS = ["bedrock/eu.anthropic.claude-sonnet-5-test", "bedrock/amazon.titan-embed-text-v2:0"]


def vector(text: str, dim: int = 128) -> list[float]:
    """Bag-of-words vector: texts sharing words get similar vectors (good enough for tests)."""
    v = [0.0] * dim
    for w in re.findall(r"[a-z]{3,}", text.lower()):
        v[int(hashlib.md5(w[:6].encode()).hexdigest(), 16) % dim] += 1
    return v


def storyboard_answer(prompt: str) -> dict:
    """A valid storyboard that uses the labels found in the prompt."""
    s = sorted(set(re.findall(r"\[(S\d+)\]", prompt)), key=lambda x: int(x[1:])) or ["S1"]
    p = sorted(set(re.findall(r"\[(P\d+)\]", prompt)), key=lambda x: int(x[1:]))
    target = int(re.search(r"about (\d+) spoken words", prompt).group(1))
    per = max(12, target // 5)
    say = ("A join combines two tables on a shared key so you see related data together "
           "and avoid copying values by hand between sheets. ") * 20
    words = say.split()
    scenes = [{"visual": "title", "heading": "Joining tables", "narration": " ".join(words[:per])}]
    if p:
        scenes.append({"visual": "slide", "slide": p[0], "narration": " ".join(words[per:2 * per]), "sources": s[:1]})
    scenes += [
        {"visual": "concept", "heading": "Join key", "items": ["A column both tables share"],
         "narration": " ".join(words[:per]), "sources": s[:1]},
        {"visual": "comparison", "heading": "Inner vs left", "columns": [
            {"heading": "Inner", "items": ["Only matches"]}, {"heading": "Left", "items": ["All of the first table"]}],
         "narration": " ".join(words[:per]), "sources": s[:1]},
        {"visual": "table", "heading": "Which join", "table_header": ["Need", "Join"],
         "table_rows": [["Only matches", "Inner"], ["Keep all rows", "Left"]],
         "narration": " ".join(words[:per]), "sources": s[:1]},
        {"visual": "steps", "heading": "How to join", "items": ["Pick the key", "Clean the key", "Join"],
         "narration": " ".join(words[:per]), "sources": s[:1]},
        {"visual": "bullets", "heading": "Key takeaways", "items": ["Use a clean key", "Check the result"],
         "narration": " ".join(words[:per]), "sources": s[-1:]},
    ]
    return {"title": "Joining tables", "goal": "Join two tables on a shared key.", "scenes": scenes}


def answer(text: str) -> str:
    """Canned model answer for the kind of request found in the prompt."""
    if "You turn a learner's request" in text:
        ids = re.findall(r"- id (\S+):", text)
        return json.dumps({"format": "video", "topic": "joining data", "recordings": ids[:1] + ["made-up-id"],
                           "minutes": 1, "level": "wizard", "wishes": "use a comparison table"})
    if "You review a narrated video lesson" in text:
        return json.dumps({"faithful": 4, "level_fit": 5, "clarity": 4, "issues": []})
    if "spoken words" in text:
        return "```json\n" + json.dumps(storyboard_answer(text)) + "\n```"
    if "self-study lesson" in text:
        labels = sorted(set(re.findall(r"\[(S\d+)\]", text)), key=lambda x: int(x[1:])) or ["S1"]
        target = int(re.search(r"about (\d+) words in total", text).group(1))
        body = " ".join((["Tables are joined on a shared key."] * 200))
        return json.dumps({"title": "Joining tables", "goal": "Understand joins.",
                           "sections": [{"heading": "Joins", "body": " ".join(body.split()[:target - 20]),
                                         "sources": labels[:1]}],
                           "takeaways": ["Key", "Clean", "Check"]})
    if "JSON" in text:
        return json.dumps({"term": "VTT file", "definition": "Subtitles with timestamps."})
    return "OK"


class Gateway:
    """Runs the fake gateway on a free local port in a background thread."""

    def __init__(self) -> None:
        self.state = {"openai_route": True, "reject_response_format": False, "calls": []}
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):   # keep test output clean
                pass

            def _send(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self) -> bool:
                ok = self.headers.get("x-bf-vk") == KEY or self.headers.get("Authorization") == f"Bearer {KEY}"
                if not ok:
                    self._send({"error": {"message": "invalid virtual key"}}, 401)
                return ok

            def do_GET(self):
                if self._authorized():
                    if self.path == "/v1/models" and gateway.state["openai_route"]:
                        return self._send({"data": [{"id": m} for m in MODELS]})
                    self._send({"error": "not found"}, 404)

            def do_POST(self):
                if not self._authorized():
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                gateway.state["calls"].append(self.path)
                if body.get("model") not in MODELS:
                    return self._send({"error": {"message": "unknown model"}}, 404)
                if self.path == "/v1/embeddings":
                    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
                    texts = texts[:1]   # like Titan: one text per request
                    return self._send({"data": [{"index": 0, "embedding": vector(texts[0])}],
                                       "usage": {"prompt_tokens": len(texts[0]) // 4}})
                if self.path == "/anthropic/v1/messages":
                    text = body.get("system", "") + " ".join(
                        m["content"] if isinstance(m["content"], str) else "" for m in body["messages"])
                    if isinstance(body["messages"][-1]["content"], list):
                        text = "TYPE: list\nSTRUCTURE: a -> b\nKEY POINTS: test"
                    out = answer(text) if "TYPE: list" not in text else text
                    return self._send({"content": [{"type": "text", "text": out}], "stop_reason": "end_turn",
                                       "usage": {"input_tokens": 100, "output_tokens": len(out) // 4}})
                if self.path == "/v1/chat/completions" and gateway.state["openai_route"]:
                    if gateway.state["reject_response_format"] and "response_format" in body:
                        return self._send({"error": {"message": "response_format not supported"}}, 400)
                    last = body["messages"][-1]["content"]
                    if isinstance(last, list):
                        out = "TYPE: list\nSTRUCTURE: a -> b\nKEY POINTS: test"
                    else:
                        out = answer(" ".join(m["content"] for m in body["messages"] if isinstance(m["content"], str)))
                    return self._send({"choices": [{"message": {"role": "assistant", "content": out},
                                                    "finish_reason": "stop"}],
                                       "usage": {"prompt_tokens": 100, "completion_tokens": len(out) // 4}})
                self._send({"error": "route not found"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True   # never wait for a hanging request at the end
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        """Stop the server without ever blocking the test run (Windows can't interrupt a wait)."""
        stopper = threading.Thread(target=self.server.shutdown, daemon=True)
        stopper.start()
        stopper.join(timeout=5)
        self.server.server_close()
