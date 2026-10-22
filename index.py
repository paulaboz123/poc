"""
index.py - semantic search over all segments.

How it works:
- Every segment becomes one "document": recording title + description + slide text +
  what was said. Slide text matters: slide titles name the topic far more clearly than
  spoken language does.
- An embedding model turns each document into a vector (a list of numbers that encodes
  meaning). Similar meaning = vectors pointing in a similar direction.
- A search embeds the question the same way and ranks documents by cosine similarity
  (1.0 = same meaning, ~0.3 = unrelated). Stored as a NumPy array: for a few thousand
  segments, a full scan takes milliseconds, so no vector database is needed yet.

Usage:
  uv run index.py build
  uv run index.py search "budget planning process" -k 10
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import numpy as np

import config
import llm
import store

log = logging.getLogger("index")


def _prefix(kind: str) -> str:
    """nomic-embed-text was trained with these prefixes; without them, quality drops."""
    if "nomic" not in config.EMBED_MODEL:
        return ""
    return "search_query: " if kind == "query" else "search_document: "


def doc_text(seg: dict) -> str:
    """Text that stands for one segment in the search index: recording title, its
    description, the slide text on screen and what was said."""
    parts = [f"Recording: {seg['title']}"]
    if seg.get("description"):
        parts.append(f"About: {seg['description']}")
    if seg.get("slide_text"):
        parts.append(f"On screen: {seg['slide_text'][:500]}")
    parts.append(seg["text"])
    return _prefix("doc") + "\n".join(parts)


def _normalize(v: np.ndarray) -> np.ndarray:
    """Scale vectors to length 1, so a dot product equals cosine similarity."""
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def build() -> int:
    """Embed every segment of the included recordings and save the index. Returns 0 if ok."""
    segments = store.all_segments()
    if not segments:
        log.error("No segments. Run ingest.py first.")
        return 1
    ids = list(segments)
    log.info("Embedding %s segments with %s ...", len(ids), config.EMBED_MODEL)
    vectors = _normalize(np.array(llm.embed([doc_text(segments[i]) for i in ids],
                                            step="index_build"), dtype="float32"))
    store.save_index(vectors, ids, {"embed_model": config.EMBED_MODEL, "n": len(ids),
                                    "built_at": datetime.now().isoformat(timespec="seconds")})
    log.info("Index built: %s segments from %s videos", len(ids), len({segments[i]['video_id'] for i in ids}))
    return 0


def search(query: str, top_k: int = 10, video_ids: set[str] | None = None) -> list[dict]:
    """Best-matching segments: [{'score': 0.71, 'segment': {...}}, ...], best first."""
    vectors, ids, info = store.load_index()
    if info.get("embed_model") != config.EMBED_MODEL:
        raise RuntimeError(f"Index was built with {info.get('embed_model')}, .env says "
                           f"{config.EMBED_MODEL}. Rebuild: uv run index.py build")
    segments = store.all_segments()
    q = _normalize(np.array(llm.embed([_prefix("query") + query], step="search")[0], dtype="float32"))
    scores = vectors @ q
    hits = []
    for i in np.argsort(-scores):
        seg = segments.get(ids[i])
        if seg is None or (video_ids and seg["video_id"] not in video_ids):
            continue   # deleted, or excluded in the manifest since the index was built
        hits.append({"score": round(float(scores[i]), 3), "segment": seg})
        if len(hits) >= top_k:
            break
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    config.setup_logging()

    if args.cmd == "build":
        return build()
    for h in search(args.query, args.k):
        seg = h["segment"]
        snippet = " ".join(seg["text"].split())[:140]
        print(f"{h['score']:.3f}  {seg['title'][:40]:40}  {store.fmt_ts(seg['start'])}  {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
