"""Retrieval evaluation.

Runs a configuration over the stored question set and records recall@k / MRR. Because the
gold label is the chunk the question was generated from, this needs no human labelling and
can be re-run after every retrieval change."""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import asdict

from sqlalchemy import select

from library_agent.db.models import EvalQuestion, EvalResult, EvalRun
from library_agent.db.session import session_scope
from library_agent.eval.generate import SUITE_RETRIEVAL, generate_questions
from library_agent.eval.metrics import RetrievalMetrics
from library_agent.llm.ollama import Ollama
from library_agent.retrieval.pipeline import LADDER, RetrievalConfig, retrieve

KS = (1, 3, 5, 10)


async def run_config(
    config: RetrievalConfig, *, suite: str = SUITE_RETRIEVAL, limit: int = 10, persist: bool = True
) -> dict[str, float]:
    async with session_scope() as db:
        questions = list(
            (await db.execute(select(EvalQuestion).where(EvalQuestion.suite == suite))).scalars()
        )
    if not questions:
        raise SystemExit(f"no questions in suite {suite!r}; run with --generate first")

    metrics = RetrievalMetrics()
    started = time.time()
    per_question: list[tuple[EvalQuestion, int | None]] = []

    async with Ollama() as client, session_scope() as db:
        for q in questions:
            hits = await retrieve(db, q.question, config=config, client=client, limit=limit)
            rank = metrics.add(
                [h.chunk_id for h in hits],
                q.gold_chunk_id,
                [h.document_id for h in hits],
                q.gold_document_id,
                KS,
            )
            per_question.append((q, rank))

    summary = metrics.summary(KS)
    summary["seconds_per_query"] = round((time.time() - started) / max(1, len(questions)), 3)

    if persist:
        async with session_scope() as db:
            run = EvalRun(suite=suite, params=asdict(config), metrics=summary)
            db.add(run)
            await db.flush()
            for q, rank in per_question:
                db.add(EvalResult(run_id=run.id, question_id=q.id, rank=rank))
    return summary


def _print_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    cols = [f"recall@{k}" for k in KS] + ["mrr", "doc_recall", "seconds_per_query"]
    width = max(len(n) for n, _ in rows) + 2
    print(f"\n{'config':<{width}}" + "".join(f"{c:>13}" for c in cols))
    print("-" * (width + 13 * len(cols)))
    base = rows[0][1] if rows else {}
    for name, m in rows:
        line = f"{name:<{width}}"
        for c in cols:
            v = m.get(c, 0)
            line += f"{v:>13.4f}" if isinstance(v, float) else f"{v:>13}"
        print(line)
    if len(rows) > 1:
        best = max(rows, key=lambda r: r[1].get("recall@10", 0))
        print(
            f"\nbest recall@10: {best[0]} ({best[1].get('recall@10', 0):.4f}) "
            f"vs {rows[0][0]} baseline ({base.get('recall@10', 0):.4f})"
        )


async def main() -> None:
    ap = argparse.ArgumentParser(description="retrieval evaluation")
    ap.add_argument("--generate", type=int, metavar="N", help="regenerate N questions first")
    ap.add_argument("--suite", default=SUITE_RETRIEVAL)
    ap.add_argument("--config", help="run one named config from the ladder")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args()

    if args.generate:
        async with Ollama() as c, session_scope() as db:
            qs = await generate_questions(db, n=args.generate, suite=args.suite, client=c)
            print(f"generated {len(qs)} questions")

    configs = LADDER
    if args.config:
        configs = [c for c in LADDER if c.name == args.config]
        if not configs:
            raise SystemExit(f"unknown config {args.config!r}; have {[c.name for c in LADDER]}")

    rows = []
    for cfg in configs:
        summary = await run_config(
            cfg, suite=args.suite, limit=args.limit, persist=not args.no_persist
        )
        print(
            f"  {cfg.name:<24} recall@10={summary.get('recall@10', 0):.4f} "
            f"mrr={summary.get('mrr', 0):.4f} ({summary.get('seconds_per_query', 0):.3f}s/q)"
        )
        rows.append((cfg.name, summary))
    _print_table(rows)


if __name__ == "__main__":
    asyncio.run(main())
