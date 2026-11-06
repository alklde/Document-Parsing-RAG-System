#!/usr/bin/env python3
"""解析层数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

KINDS = ("title", "text", "table", "figure", "header", "footer", "list", "caption", "equation")


@dataclass
class TextBlock:
    text: str
    kind: str = "text"
    page: int = 0
    bbox: List[float] = field(default_factory=list)   # [x0,y0,x1,y1] 绝对值
    order: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_chunk_payload(self, doc_id: str) -> Dict[str, Any]:
        return {
            "doc_id": doc_id,
            "text": self.text,
            "meta": {
                "kind": self.kind,
                "page": self.page,
                "order": self.order,
                "bbox": self.bbox,
            },
        }


@dataclass
class ParsedDocument:
    source: str                 # 原始路径
    doc_type: str               # pdf / image / docx
    pages: int
    blocks: List[TextBlock]
    raw: Dict[str, Any] = field(default_factory=dict)   # 原始结果备份

    def to_json(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "pages": self.pages,
            "blocks": [
                {
                    "text": b.text,
                    "kind": b.kind,
                    "page": b.page,
                    "bbox": b.bbox,
                    "order": b.order,
                    "meta": b.meta,
                }
                for b in self.blocks
            ],
        }

    @property
    def text_blocks(self) -> List[TextBlock]:
        return [b for b in self.blocks if b.kind != "figure"]