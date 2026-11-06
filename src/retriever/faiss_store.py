#!/usr/bin/env python3
"""Faiss 向量索引：建库、持久化、检索。"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from ..config import Config
from .embedder import BgeEmbedder


class FaissStore:
    def __init__(self, cfg: Config, embedder: BgeEmbedder):
        self.cfg = cfg
        self.embedder = embedder
        self.index_dir = cfg.index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self._index_path = os.path.join(self.index_dir, "index.faiss")
        self._meta_path = os.path.join(self.index_dir, "meta.json")
        self._index = None
        self._chunks: List[Dict[str, Any]] = []
        self._load()

    # -------------------------------------------------------------- index
    def _load(self):
        import faiss

        if os.path.exists(self._index_path):
            self._index = faiss.read_index(self._index_path)
        if os.path.exists(self._meta_path):
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._chunks = json.load(f)

    def _ensure_index(self, dim: int):
        import faiss

        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
        # 维度不一致时报错，提示重建
        if self._index.d != dim:
            raise ValueError(f"索引维度 {self._index.d} 与当前嵌入维度 {dim} 不一致，请用 --rebuild 重建")

    def save(self):
        import faiss

        faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)

    def rebuild(self, chunks: List[Dict[str, Any]]):
        """全量重建索引。"""
        import faiss

        self._chunks = []
        self._index = None
        self.add(chunks)
        self.save()

    def add(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        vecs = self.embedder.embed_documents(texts)
        self._ensure_index(vecs.shape[1])
        self._index.add(vecs)
        self._chunks.extend(chunks)
        self.save()

    # ------------------------------------------------------------- search
    def search(self, query: str, k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        qv = self.embedder.embed_query(query).reshape(1, -1)
        scores, ids = self._index.search(qv, min(k, self._index.ntotal))
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            hits.append((self._chunks[int(idx)], float(score)))
        return hits

    def count(self) -> int:
        return 0 if self._index is None else self._index.ntotal