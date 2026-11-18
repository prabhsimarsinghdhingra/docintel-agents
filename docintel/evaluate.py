"""Evaluation pipeline -> dashboard/data/results.json
Metrics: retrieval hit@k (expected doc among cited evidence), top-1 doc hit,
answer keyword coverage, grounded-claim ratio, latency."""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

from .agents import Pipeline
from .embeddings import get_embedder
from .index import VectorIndex


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--questions", default="evaluation/questions.json")
    ap.add_argument("--out", default="dashboard/data/results.json")
    ap.add_argument("--require-llm", action="store_true", help="fail if any answer fell back to extractive")
    ap.add_argument("--min-hit", type=float, default=0.7)
    ap.add_argument("--min-grounded", type=float, default=0.7)
    a = ap.parse_args()

    idx = VectorIndex.load(a.index, get_embedder())
    pipe = Pipeline(idx)
    rows = []
    for q in json.loads(Path(a.questions).read_text()):
        r = pipe.ask(q["question"])
        docs = [c["doc"].lower() for c in r["citations"]]
        exp = q["expected_doc"].lower()
        hit = any(exp in d for d in docs)
        top1 = bool(docs) and exp in docs[0]
        kw = any(k.lower() in r["answer"].lower() for k in q["keywords_any"])
        rows.append({**r, "id": q["id"], "expected_doc": q["expected_doc"],
                     "retrieval_hit": hit, "top1_hit": top1, "keyword_hit": kw,
                     "grounded_ratio": r["verification"]["grounded_ratio"]})
        print(f"{q['id']}: hit={hit} top1={top1} kw={kw} grounded={rows[-1]['grounded_ratio']}")

    n = len(rows)
    lat = sorted(r["latency_s"] for r in rows)
    summary = {
        "questions": n,
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in rows) / n, 3),
        "top1_hit_rate": round(sum(r["top1_hit"] for r in rows) / n, 3),
        "keyword_hit_rate": round(sum(r["keyword_hit"] for r in rows) / n, 3),
        "avg_grounded_ratio": round(st.mean(r["grounded_ratio"] for r in rows), 3),
        "verified_rate": round(sum(r["status"] == "verified" for r in rows) / n, 3),
        "llm_fallback_rate": round(sum(r["llm_fallback"] for r in rows) / n, 3),
        "retry_rate": round(sum(r["attempts"] > 1 for r in rows) / n, 3),
        "latency_p50_s": lat[n // 2],
        "latency_max_s": lat[-1],
    }
    modes = {r["mode"] for r in rows}
    mode = modes.pop() if len(modes) == 1 else "mixed"
    if summary["llm_fallback_rate"] > 0:
        print(f"WARNING: {summary['llm_fallback_rate']:.0%} of answers fell back to extractive "
              "because the Gemini call failed. See each result's trace for the error.")
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "mode": mode,
           "embedder": idx.embedder.name,
           "documents": sorted({c.doc for c in idx.chunks}),
           "chunks": len(idx.chunks), "summary": summary, "results": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    ok = summary["retrieval_hit_rate"] >= a.min_hit and summary["avg_grounded_ratio"] >= a.min_grounded
    if a.require_llm and (not pipe.llm.available or summary["llm_fallback_rate"] > 0):
        print("FAIL: --require-llm set but LLM answers were not produced for every question")
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
