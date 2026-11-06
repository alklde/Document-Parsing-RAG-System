#!/usr/bin/env python3
"""CLI 入口：parse / index / ask / eval。

用法（在项目根目录）：
    python -m src.cli parse <file> [-o out.json]
    python -m src.cli index <file|dir> [--rebuild]
    python -m src.cli ask "<问题>" [--k 4]
    python -m src.cli eval [--bench data/benchmark.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    from src.config import Config

    cfg = Config.load(os.path.join(ROOT, "config.yaml"), root=ROOT)

    parser = argparse.ArgumentParser(prog="docagent", description="智能文档解析与多模态 RAG 问答")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="仅解析文档并输出 JSON")
    p_parse.add_argument("path")
    p_parse.add_argument("-o", "--out", default=None)

    p_index = sub.add_parser("index", help="解析+嵌入+建索引")
    p_index.add_argument("path", help="文件或目录(默认 data/source)")
    p_index.add_argument("--rebuild", action="store_true", help="重建索引")
    p_index.add_argument("--fast", action="store_true", help="用随机向量占位, 验证链路")

    p_ask = sub.add_parser("ask", help="基于索引问答")
    p_ask.add_argument("question")
    p_ask.add_argument("--k", type=int, default=int(cfg.get("llm", "top_k", 4)))
    p_ask.add_argument("--model", default=None, help="覆盖 LLM 模型")

    p_eval = sub.add_parser("eval", help="运行检索/问答评测")
    p_eval.add_argument("--bench", default=os.path.join(ROOT, "data", "benchmark.json"))
    p_eval.add_argument("--k", type=int, default=int(cfg.get("llm", "top_k", 4)))
    p_eval.add_argument("--with-llm", action="store_true", help="同时跑关键词判分")

    args = parser.parse_args()
    if args.cmd == "parse":
        return _cmd_parse(args, cfg)
    if args.cmd == "index":
        return _cmd_index(args, cfg)
    if args.cmd == "ask":
        return _cmd_ask(args, cfg)
    if args.cmd == "eval":
        return _cmd_eval(args, cfg)
    return 1


def _cmd_parse(args, cfg) -> int:
    from src.document_parser.pipeline import DocumentParser

    dp = DocumentParser(cfg)
    parsed = dp.parse(args.path)
    js = parsed.to_json()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False, indent=2)
        print(f"已输出 {args.out}")
    else:
        print(json.dumps(js, ensure_ascii=False, indent=2)[:4000])
    print(f"[block={len(parsed.blocks)} pages={parsed.pages}]")
    return 0


def _cmd_index(args, cfg) -> int:
    from src.document_parser.pipeline import DocumentParser
    from src.retriever.chunker import build_chunks
    from src.retriever.embedder import BgeEmbedder, DummyEmbedder
    from src.retriever.faiss_store import FaissStore

    path = args.path
    dp = DocumentParser(cfg)
    if os.path.isdir(path):
        docs = dp.parse_dir(path)
    elif os.path.isfile(path):
        docs = [dp.parse(path)]
    else:
        docs = [dp.parse(os.path.join(cfg.source_dir, path))]

    # 释放 OCR 模型显存，给 bge 嵌入腾空间（8GB 显存较紧张）
    del dp
    import gc

    gc.collect()
    try:
        import paddle

        paddle.device.cuda.empty_cache()
    except Exception:
        pass

    chunks = build_chunks(docs, cfg)
    embedder = DummyEmbedder(cfg) if args.fast else BgeEmbedder(cfg)
    store = FaissStore(cfg, embedder)
    if args.rebuild:
        store.rebuild(chunks)
    else:
        store.add(chunks)

    print(f"解析 {len(docs)} 个文档 -> {len(chunks)} chunks -> 索引 {store.count()} 条")
    return 0


def _cmd_ask(args, cfg) -> int:
    import time

    from src.retriever.embedder import BgeEmbedder, DummyEmbedder
    from src.retriever.faiss_store import FaissStore
    from src.rag.generator import RagGenerator

    if not os.path.exists(os.path.join(cfg.index_dir, "index.faiss")):
        print("索引为空，请先运行: python -m src.cli index data/source")
        return 1

    # fast 索引(随机向量)也能问, 只是检索质量差; 有 index 即可
    embedder = BgeEmbedder(cfg)
    store = FaissStore(cfg, embedder)
    if store.count() == 0:
        print("索引为空")
        return 1

    gen = RagGenerator(cfg)
    if args.model:
        gen.llm.model = args.model

    hits = store.search(args.question, k=args.k)
    print(f"[召回 {len(hits)} 条]")
    for i, (c, s) in enumerate(hits, 1):
        m = c["meta"]
        print(f"  [{i}] score={s:.3f} {m.get('source')}:p{m.get('page',0)+1}({m.get('kind')}) {c['text'][:60]}...")

    t0 = time.perf_counter()
    result = gen.answer(args.question, hits)
    print(f"\n[模型] {result['model']} ({time.perf_counter()-t0:.1f}s)\n")
    print(result["answer"])
    print("\n[引用来源]")
    for s in result["sources"][:args.k]:
        print(f"  - {s['source']} 第{s['page']}页 ({s['kind']}, score={s['score']})")
    return 0


def _cmd_eval(args, cfg) -> int:
    from src.eval.bench import load_bench, run_retrieval_eval
    from src.retriever.embedder import BgeEmbedder
    from src.retriever.faiss_store import FaissStore

    bench = load_bench(args.bench)
    store = FaissStore(cfg, BgeEmbedder(cfg))
    result = run_retrieval_eval(store, bench, k=args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.with_llm:
        from src.eval.bench import run_answer_eval
        from src.rag.generator import RagGenerator

        r = run_answer_eval(RagGenerator(cfg), store, bench, k=args.k)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())