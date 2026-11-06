#!/usr/bin/env python3
"""RAG 问答组装。"""
from __future__ import annotations

from typing import Any, Dict, List

from ..config import Config


class RagGenerator:
    def __init__(self, cfg: Config):
        from .llm import make_llm_backend

        self.cfg = cfg
        self.llm = make_llm_backend(cfg)
        self.llm_name = self.llm.model

    def answer(
        self,
        question: str,
        chunks: List[tuple],
    ) -> Dict[str, Any]:
        from .llm import build_prompt

        messages = build_prompt(question, chunks)
        content = self.llm.chat([dict(m) for m in messages])
        sources = [
            {
                "source": c["meta"].get("source", ""),
                "page": c["meta"].get("page", 0) + 1,
                "kind": c["meta"].get("kind", "text"),
                "text": c["text"][:120],
                "score": round(s, 4),
            }
            for c, s in chunks
        ]
        return {"answer": content, "sources": sources, "model": self.llm_name}