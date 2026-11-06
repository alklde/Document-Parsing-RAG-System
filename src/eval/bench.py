#!/usr/bin/env python3
"""小型评测：检索命中率(Recall@k)、MRR、问答关键词命中。"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ..config import Config
from ..retriever.faiss_store import FaissStore


def _hr(question: str, store: FaissStore, k: int = 4) -> List[Dict[str, Any]]:
    hits = store.search(question, k=k)
    return [c for c, _ in hits]


def _passage_hit(chunk: Dict[str, Any], passage: str) -> bool:
    return passage in chunk["text"]


def run_retrieval_eval(store: FaissStore, bench: List[Dict[str, Any]], k: int = 4) -> Dict[str, Any]:
    """bench 元素: {question, passage}。passage 为应命中的原文片段。"""
    recall_at_k = 0.0
    mrr = 0.0
    n = 0
    per: List[Dict[str, Any]] = []
    total_ms = 0.0
    import time

    for item in bench:
        q = item["question"]
        passage = item.get("passage", "")
        t0 = time.perf_counter()
        chunks = _hr(q, store, k=k)
        total_ms += (time.perf_counter() - t0) * 1000
        n += 1
        hit_i = next((i for i, c in enumerate(chunks) if _passage_hit(c, passage)), None)
        if hit_i is not None:
            recall_at_k += 1
            mrr += 1.0 / (hit_i + 1)
        per.append({"question": q, "hit": hit_i is not None, "rank": None if hit_i is None else hit_i + 1})

    return {
        "samples": n,
        "recall_at_k": round(recall_at_k / n, 4) if n else 0.0,
        "mrr": round(mrr / n, 4) if n else 0.0,
        "avg_search_ms": round(total_ms / n, 2) if n else 0.0,
        "cases": per,
    }


def run_answer_eval(generator, store: FaissStore, bench: List[Dict[str, Any]], k: int = 4) -> Dict[str, Any]:
    """答案关键词命中率（确定性判分，不依赖裁判模型）。"""
    ok = 0
    n = 0
    per: List[Dict[str, Any]] = []
    for item in bench:
        q = item["question"]
        keywords = item.get("keywords", [])
        if not keywords:
            continue
        chunks = _hr(q, store, k=k)
        if not chunks:
            per.append({"question": q, "hit": False, "note": "无召回"})
            continue
        n += 1
        result = generator.answer(q, [(c, 1.0) for c in chunks[:k]])
        answer = result["answer"]
        hit = all(kw in answer for kw in keywords)
        ok += int(hit)
        per.append({"question": q, "hit": hit, "answer": answer[:80]})
    return {
        "samples": n,
        "keyword_hit_rate": round(ok / n, 4) if n else 0.0,
        "cases": per,
    }


def load_bench(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_bench_from_docs(parsed_docs) -> List[Dict[str, Any]]:
    """从解析结果自动采样少量 (question, passage) 生成轻量基准集。"""
    raise NotImplementedError("建议人工编写 data/benchmark.json 以保证可用性")