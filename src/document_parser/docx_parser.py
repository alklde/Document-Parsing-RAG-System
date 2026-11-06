#!/usr/bin/env python3
"""Word(.docx) 解析：原生段落/标题/表格，跳过内嵌图片的 OCR。"""
from __future__ import annotations

import os
from typing import List

from ..config import Config
from .models import ParsedDocument, TextBlock


def _table_to_markdown(table) -> str:
    rows: List[List[str]] = []
    for row in table.rows:
        cells = [" ".join(c.text.split()) for c in row.cells]
        rows.append(cells)
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    out: List[str] = []
    for i, row in enumerate(rows):
        cells = row + [""] * (ncols - len(row))
        out.append("| " + " | ".join(cells) + " |")
        if i == 0:
            out.append("|" + "---|" * ncols)
    return "\n".join(out)


def parse_docx(path: str, cfg: Config, ocr: "PaddleOcrBackend") -> ParsedDocument:
    from docx import Document

    doc = Document(path)
    blocks: List[TextBlock] = []
    order = 0

    for para in doc.paragraphs:
        text = " ".join(para.text.split())
        if not text:
            continue
        style = (para.style.name or "").lower()
        kind = "title" if ("head" in style or "标题" in style or style == "title") else "text"
        blocks.append(TextBlock(text=text, kind=kind, page=0, order=order, meta={"style": para.style.name}))
        order += 1

    for t_i, table in enumerate(doc.tables):
        md = _table_to_markdown(table)
        if md:
            blocks.append(TextBlock(text=md, kind="table", page=0, order=order, meta={"table_idx": t_i}))
            order += 1

    return ParsedDocument(source=path, doc_type="docx", pages=1, blocks=blocks)