"""LangGraph multi-agent pipeline: planner -> researcher -> analyst -> verifier,
with a verification-driven retry loop back to the researcher."""
from __future__ import annotations

import json
import re
import time
from typing import TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from .ingest import split_sentences
from .llm import LLM
from .tools import Toolbox

CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
GROUND_T = 0.7      # min fraction of supported claims
SIM_T = 0.45        # cosine threshold claim<->cited chunk
OVERLAP_T = 0.6     # content-word overlap threshold
MAX_ATTEMPTS = 2


class State(TypedDict, total=False):
    question: str
    sub_queries: list[str]
    evidence: list[dict]
    answer: str
    verification: dict
    attempts: int
    k: int
    status: str
    mode: str
    trace: list[str]


def _toks(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3}


def build_graph(tools: Toolbox, llm: LLM, embedder):
    def planner(s: State) -> State:
        q = s["question"]
        subs = [q]
        note = ""
        if llm.available:
            try:
                raw = llm.complete(
                    'Decompose the question into at most 3 short search queries for a '
                    'document index. Return only JSON: {"sub_queries": [..]}', q, 200)
                raw = raw.strip().strip("`").removeprefix("json").strip()
                subs += [x for x in json.loads(raw)["sub_queries"] if isinstance(x, str)][:3]
            except Exception as exc:
                note = f" (LLM failed: {str(exc)[:160]})"
        else:
            parts = [p.strip() for p in re.split(r"\?|\band\b|;", q) if len(p.split()) >= 3]
            subs += [p for p in parts if p.lower() != q.lower()][:2]
        return {"sub_queries": subs, "trace": s.get("trace", []) + [f"planner: {len(subs)} sub-queries{note}"]}

    def researcher(s: State) -> State:
        attempts = s.get("attempts", 0) + 1
        k = 5 + 4 * (attempts - 1)
        merged: dict[int, dict] = {}
        for sq in s["sub_queries"]:
            for h in tools.search_documents(sq, k):
                cur = merged.get(h["chunk_id"])
                if cur is None or h["score"] > cur["score"]:
                    merged[h["chunk_id"]] = h
        ev = sorted(merged.values(), key=lambda h: -h["score"])[: k + 3]
        ev = [{**h, "ref": i + 1} for i, h in enumerate(ev)]
        return {"evidence": ev, "attempts": attempts, "k": k,
                "trace": s.get("trace", []) + [f"researcher: attempt {attempts}, k={k}, {len(ev)} chunks"]}

    def analyst(s: State) -> State:
        ev, q = s["evidence"], s["question"]
        ans, mode, note = None, "extractive", ""
        if llm.available:
            ctx = "\n\n".join(f"[{e['ref']}] ({e['doc']} p.{e['page']}) {e['text']}" for e in ev)
            try:
                ans = llm.complete(
                    "Answer using ONLY the numbered evidence. End every sentence with its "
                    "citation like [2] or [1,3]. If the evidence is insufficient, say "
                    "'Insufficient evidence.' Be concise (max 5 sentences).",
                    f"Evidence:\n{ctx}\n\nQuestion: {q}")
                mode = "llm"
            except Exception as exc:  # fall back so the pipeline never dies on API limits
                note = f" (LLM failed: {str(exc)[:160]}; fell back)"
        if ans is None:
            cands = [(e["ref"], x) for e in ev for x in split_sentences(e["text"]) if 40 <= len(x) <= 400]
            if not cands:
                ans = "Insufficient evidence."
            else:
                v = embedder.encode([q] + [x for _, x in cands])
                order = np.argsort(-(v[1:] @ v[0]))
                picked, seen = [], set()
                for i in order:
                    ref, x = cands[i]
                    key = x[:60].lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    picked.append(f"{x.rstrip('.!?')} [{ref}].")
                    if len(picked) == 3:
                        break
                ans = " ".join(picked)
        return {"answer": ans.strip(), "mode": mode,
                "trace": s.get("trace", []) + [f"analyst: {mode} answer{note}"]}

    def verifier(s: State) -> State:
        ans = re.sub(r"([.!?])\s+(\[\d[\d,\s]*\])", r" \2\1", s["answer"])
        claims = split_sentences(ans)
        by_ref = {e["ref"]: e["text"] for e in s["evidence"]}
        ref_ids = list(by_ref)
        ev_vecs = embedder.encode([by_ref[r] for r in ref_ids]) if ref_ids else None
        claim_vecs = embedder.encode([CITE.sub("", c) for c in claims]) if claims else None
        results = []
        insufficient = s["answer"].lower().startswith("insufficient")
        for i, c in enumerate(claims):
            refs = sorted({int(n) for m in CITE.findall(c) for n in re.split(r"\s*,\s*", m)
                           if int(n) in by_ref})
            text = CITE.sub("", c).strip()
            sim = overlap = 0.0
            if refs:
                idx = [ref_ids.index(r) for r in refs]
                sim = float(max(ev_vecs[j] @ claim_vecs[i] for j in idx))
                ct = _toks(text)
                joined = _toks(" ".join(by_ref[r] for r in refs))
                overlap = len(ct & joined) / len(ct) if ct else 0.0
            ok = bool(refs) and (sim >= SIM_T or overlap >= OVERLAP_T)
            results.append({"claim": text, "refs": refs, "similarity": round(sim, 3),
                            "overlap": round(overlap, 3), "supported": ok})
        ratio = 0.0 if (insufficient or not results) else sum(r["supported"] for r in results) / len(results)
        status = "verified" if ratio >= GROUND_T else "low_confidence"
        return {"verification": {"grounded_ratio": round(ratio, 3), "claims": results},
                "status": status,
                "trace": s.get("trace", []) + [f"verifier: grounded={ratio:.2f} -> {status}"]}

    def route(s: State) -> str:
        if s["verification"]["grounded_ratio"] >= GROUND_T or s.get("attempts", 0) >= MAX_ATTEMPTS:
            return END
        return "researcher"

    g = StateGraph(State)
    for name, fn in [("planner", planner), ("researcher", researcher),
                     ("analyst", analyst), ("verifier", verifier)]:
        g.add_node(name, fn)
    g.set_entry_point("planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "verifier")
    g.add_conditional_edges("verifier", route, {"researcher": "researcher", END: END})
    return g.compile()


class Pipeline:
    def __init__(self, index, llm: LLM | None = None):
        self.index = index
        self.llm = llm or LLM()
        self.graph = build_graph(Toolbox(index), self.llm, index.embedder)

    def ask(self, question: str) -> dict:
        t0 = time.time()
        s = self.graph.invoke({"question": question})
        return {
            "question": question,
            "answer": s["answer"],
            "status": s["status"],
            "mode": s["mode"],
            "llm_fallback": self.llm.available and s["mode"] != "llm",
            "sub_queries": s["sub_queries"],
            "attempts": s["attempts"],
            "citations": [{"ref": e["ref"], "doc": e["doc"], "page": e["page"],
                           "score": e["score"], "text": e["text"][:500]} for e in s["evidence"]],
            "verification": s["verification"],
            "trace": s["trace"],
            "latency_s": round(time.time() - t0, 3),
        }
