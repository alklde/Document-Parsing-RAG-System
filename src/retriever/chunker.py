#!/usr/bin/env python3
"""文本分块：按版面阅读顺序切分，保留引用元数据。"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from ..config import Config
from ..document_parser.models import ParsedDocument

_RETRIEVABLE = {"title", "text", "table", "list", "caption"}
_SKIP = {"figure", "header", "footer", "equation"}


def doc_id_of(source: str) -> str:
    return os.path.splitext(os.path.basename(source))[0]


def build_chunks(docs: List[ParsedDocument], cfg: Config) -> List[Dict[str, Any]]:
    size = int(cfg.get("chunk", "size", 400))
    overlap = int(cfg.get("chunk", "overlap", 80))
    min_len = int(cfg.get("chunk", "min_chunk_chars", 20))

    chunks: List[Dict[str, Any]] = []
    for doc in docs:
        doc_id = doc_id_of(doc.source)
        for i, blk in enumerate(doc.blocks):
            if blk.kind in _SKIP:
                continue

            meta = {
                "doc_id": doc_id,
                "source": doc.source,
                "page": blk.page,
                "kind": blk.kind,
                "order": i,
            }
            # 表格单独成块，完整保留行列信息
            if blk.kind == "table":
                if len(blk.text) >= min_len:
                    chunks.append({"text": blk.text, "meta": meta})
                continue

            # 文本块做语义分块：按块顺序拼接至 size，允许 overlap
            text = blk.text
            if not text:
                continue
            if len(text) <= size:
                if len(text) >= min_len:
                    chunks.append({"text": text, "meta": meta})
                continue
            step = size - overlap
            start = 0
            while start < len(text):
                seg = text[start:start + size]
                seg_meta = dict(meta)
                seg_meta["offset"] = start
                if len(seg) >= min_len:
                    chunks.append({"text": seg, "meta": seg_meta})
                start += step

    return chunks