"""Does tuning a BM25-vs-dense weight inside RRF help? Tune on dev, check on test.

Weighted RRF: score(d) = w / (60 + rank_bm25(d)) + (1 - w) / (60 + rank_dense(d)).
w = 0.5 is ordinary RRF, w = 1 is BM25 only, w = 0 is dense only.
Output: docs/proof/rrf_weight_experiment.txt
"""

import statistics
from pathlib import Path

from raglens.config import Settings
from raglens.embed.factory import build_embedder
from raglens.eval.dataset import load_dataset
from raglens.eval.metrics import doc_ranking, reciprocal_rank
from raglens.pipeline.builder import build_retriever, load_chunks
from raglens.retrieve.base import SearchResult

WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def fuse(bm25: list[SearchResult], dense: list[SearchResult], w: float) -> list[SearchResult]:
    """Weighted reciprocal rank fusion of two chunk rankings."""
    scores: dict[str, float] = {}
    by_id = {r.chunk.chunk_id: r for r in bm25 + dense}
    for results, weight in ((bm25, w), (dense, 1 - w)):
        for r in results:
            scores[r.chunk.chunk_id] = scores.get(r.chunk.chunk_id, 0.0) + weight / (60 + r.rank)
    return [by_id[cid] for cid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def main() -> None:
    """Sweep the weight for each embedder at chunk size 512 and report dev-chosen vs test."""
    settings = Settings()
    items = [i for i in load_dataset("data/eval/qa.jsonl") if i.answerable]
    lines = ["Weighted RRF: MRR@10 on dev / test (answerable hand-labelled questions)", ""]
    for emb in ("glove", "lsa"):
        chunks = load_chunks(settings.corpus_dir, 512, 64)
        retriever = build_retriever(chunks, build_embedder(emb), settings)
        cache = {
            i.id: (
                retriever.bm25_retriever.retrieve(i.question, 40),
                retriever.dense_retriever.retrieve(i.question, 40),
            )
            for i in items
        }

        def mrr(
            w: float,
            split: str,
            cache: dict[str, tuple[list[SearchResult], list[SearchResult]]] = cache,
        ) -> float:
            return statistics.fmean(
                reciprocal_rank(doc_ranking(fuse(*cache[i.id], w)), set(i.relevant_doc_ids))
                for i in items
                if i.split == split
            )

        dev = {w: mrr(w, "dev") for w in WEIGHTS}
        test = {w: mrr(w, "test") for w in WEIGHTS}
        best_w = max(WEIGHTS, key=lambda w: dev[w])
        lines.append(f"embedder={emb}")
        lines.append("  w    " + "  ".join(f"{w:>5.1f}" for w in WEIGHTS))
        lines.append("  dev  " + "  ".join(f"{dev[w]:.3f}" for w in WEIGHTS))
        lines.append("  test " + "  ".join(f"{test[w]:.3f}" for w in WEIGHTS))
        lines.append(
            f"  dev-chosen w={best_w:.1f}: test {test[best_w]:.3f} vs plain RRF (w=0.5) "
            f"{test[0.5]:.3f}; best possible on test {max(test.values()):.3f}"
        )
        lines.append("")
    text = "\n".join(lines)
    print(text)
    out = Path("docs/proof/rrf_weight_experiment.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
