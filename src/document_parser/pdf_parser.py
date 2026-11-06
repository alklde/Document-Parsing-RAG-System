#!/usr/bin/env python3
"""PDF 解析：优先文字层，扫描页降级 OCR。"""
from __future__ import annotations

import unicodedata
from typing import List, Tuple

import pymupdf

from ..config import Config
from .models import ParsedDocument, TextBlock
from .ocr_backend import PaddleOcrBackend


def normalize_text(text: str) -> str:
    """NFKC 规范化：修复 CJK 兼容字（如 '年'->'年'）、全角/半角差异。"""
    return unicodedata.normalize("NFKC", text)


def _text_density(page: pymupdf.Page) -> float:
    try:
        txt = page.get_text("text")
    except Exception:
        return 0.0
    return len(txt.strip())


def parse_pdf(path: str, cfg: Config, ocr: PaddleOcrBackend) -> ParsedDocument:
    doc = pymupdf.open(path)
    blocks: List[TextBlock] = []

    for pno in range(len(doc)):
        page = doc[pno]
        density = _text_density(page)
        # 有文字层直接用文字层（快+准+无噪声）；纯图片页（扫描件）才走 OCR/版面
        if density > 0:
            blocks.extend(_text_layer(page, pno))
        else:
            pix = page.get_pixmap(dpi=200)
            img = _pix_to_numpy(pix)
            for b in ocr.analyze(img, page=pno):
                b.text = normalize_text(b.text)
                blocks.append(b)

    max_page = len(doc)
    doc.close()
    return ParsedDocument(source=path, doc_type="pdf", pages=max_page, blocks=blocks)


def _iter_lines(page: pymupdf.Page) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
    """按行抽取：返回 (行文本, 行内最大字号, bbox)。"""
    lines: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    data = page.get_text("dict")
    for blk in data.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            max_size = max(float(s.get("size", 0) or 0) for s in spans)
            text = normalize_text("".join(s.get("text", "") for s in spans)).strip()
            if not text:
                continue
            bbox = tuple(float(v) for v in line.get("bbox", (0, 0, 0, 0)))
            lines.append((text, max_size, bbox))
    return lines


_PARA_TERMINATORS = set("。！？；：…）】\"") 
_TABLE_CELL = {"\t", "\x00"}


def _reflow_paragraphs(lines: List[str]) -> List[str]:
    """把 PDF 文字层按行抽取的内容重排成段落。

    PDF 一“行”常是版面折行而非语义停顿，若上一段未以句末标点结尾则续行合并；含制表符/\x00 的行视为表格行，单独成段。
    """
    out: List[str] = []
    for raw in lines:
        ln = raw.strip()
        if not ln:
            continue
        if out and not any(ch in ln for ch in _TABLE_CELL) \
           and ln[-1] not in _TABLE_CELL \
           and out[-1][-1] not in _PARA_TERMINATORS \
           and not any(ch in out[-1] for ch in _TABLE_CELL):
            out[-1] += ln
        else:
            out.append(ln)
    return out


def _text_layer(page: pymupdf.Page, pno: int) -> List[TextBlock]:
    """按行解析，逐行判定标题（字号 ≥ 1.2×正文基准），标题单独成块。

    正文基准字号取「字符数加权众数」：正文行通常最长，避免中位数落在标题上。
    """
    from collections import Counter

    lines = _iter_lines(page)
    if not lines:
        return []
    total_chars: Counter = Counter()
    for text, size, _ in lines:
        total_chars[size] += len(text)
    body_baseline = total_chars.most_common(1)[0][0]
    title_threshold = body_baseline * 1.2

    blocks: List[TextBlock] = []
    buf_text: List[str] = []
    buf_bbox: List[Tuple[float, float, float, float]] = []

    def flush(order_ref: List[int]):
        nonlocal buf_text, buf_bbox
        if buf_text:
            b0 = [min(v[0] for v in buf_bbox), min(v[1] for v in buf_bbox),
                  max(v[2] for v in buf_bbox), max(v[3] for v in buf_bbox)]
            blocks.append(
                TextBlock(text="\n".join(_reflow_paragraphs(buf_text)), kind="text", page=pno,
                          bbox=b0, order=order_ref[0], meta={"lines": len(buf_text)})
            )
            order_ref[0] += 1
            buf_text, buf_bbox = [], []

    order = [0]
    for text, size, bbox in lines:
        if size >= title_threshold and len(text) < 80:
            flush(order)
            blocks.append(
                TextBlock(text=text, kind="title", page=pno,
                          bbox=list(bbox), order=order[0], meta={"font_size": round(size, 1)})
            )
            order[0] += 1
        else:
            buf_text.append(text)
            buf_bbox.append(bbox)
    flush(order)
    return blocks


def _pix_to_numpy(pix: pymupdf.Pixmap):
    import numpy as np

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    return img.copy()