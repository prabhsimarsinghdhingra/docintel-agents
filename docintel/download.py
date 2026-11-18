"""Download real research papers from arXiv into data/docs."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

PAPERS = {
    "attention_is_all_you_need.pdf": "1706.03762",
    "rag_lewis_2020.pdf": "2005.11401",
    "bert.pdf": "1810.04805",
    "sentence_bert.pdf": "1908.10084",
    "faiss_billion_scale.pdf": "1702.08734",
    "react.pdf": "2210.03629",
    "self_rag.pdf": "2310.11511",
    "lost_in_the_middle.pdf": "2307.03172",
    "toolformer.pdf": "2302.04761",
}


def main(out: str = "data/docs") -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    for name, arx in PAPERS.items():
        dest = Path(out) / name
        if dest.exists() and dest.stat().st_size > 10_000:
            print("have", name)
            continue
        req = urllib.request.Request(f"https://arxiv.org/pdf/{arx}",
                                     headers={"User-Agent": "docintel/1.0"})
        print("downloading", name)
        dest.write_bytes(urllib.request.urlopen(req, timeout=60).read())


if __name__ == "__main__":
    main(*sys.argv[1:])
