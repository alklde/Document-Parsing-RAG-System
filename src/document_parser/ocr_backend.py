#!/usr/bin/env python3
"""PaddleOCR PP-StructureV3 后端。

对图像做 OCR + 版面分析 + 表格结构还原，返回结构化 TextBlock 列表。
解耦入口：analyze(image) -> list[TextBlock]。
"""
from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..config import Config
from .models import TextBlock

# PP-StructureV3 标签 -> 语义 kind
_LABEL_MAP = {
    "doc_title": "title",
    "title": "title",
    "paragraph_title": "title",
    "text": "text",
    "list_item": "list",
    "paragraph": "text",
    "abstract": "text",
    "table": "table",
    "figure": "figure",
    "figure_caption": "caption",
    "table_caption": "caption",
    "caption": "caption",
    "header": "header",
    "footer": "footer",
    "page_number": "footer",
    "formula": "equation",
    "equation": "equation",
}


def _html_to_markdown(html: str) -> str:
    """将 PP-Structure 输出的 <table> HTML 转为 Markdown 管道表格。"""
    from html.parser import HTMLParser

    class T(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.rows: List[List[str]] = []
            self.cur_row: List[str] = []
            self.cur_cell: List[str] = []

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            if tag == "tr":
                self.cur_row = []
            elif tag in ("td", "th"):
                self.cur_cell = []

        def handle_endtag(self, tag: str) -> None:
            if tag in ("td", "th"):
                self.cur_row.append(" ".join("".join(value for value in self.cur_cell).split()))
            elif tag == "tr" and self.cur_row:
                self.rows.append(self.cur_row)

        def handle_data(self, data: str) -> None:
            if getattr(self, "cur_cell", None) is not None:
                self.cur_cell.append(data)

    p = T()
    try:
        p.feed(html)
    except Exception:
        return ""
    rows = [r for r in p.rows if r]
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


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [str(value)]


class PaddleOcrBackend:
    """PP-StructureV3 版面解析后端（OCR + 版面 + 表格）。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._engine: Any = None

    @staticmethod
    def _device_kwargs(settings: Dict[str, Any]) -> Dict[str, Any]:
        """device=auto 时不传参（走 Paddle 默认）；cpu 时禁用 oneDNN，绕开 Paddle PIR 算子兼容 bug。"""
        device = str(settings.get("device", "auto")).lower()
        if device == "cpu":
            return {"device": "cpu", "enable_mkldnn": False}
        if device == "gpu" or device.startswith("cuda"):
            return {"device": device}
        return {}

    def _ensure(self) -> Any:
        if self._engine is None:
            from paddleocr import PPStructureV3

            settings = self.cfg.section("ocr")
            self._engine = PPStructureV3(
                lang=settings.get("lang", "ch"),
                use_doc_orientation_classify=settings.get("use_doc_orientation", False),
                use_doc_unwarping=settings.get("use_doc_unwarping", False),
                use_textline_orientation=settings.get("use_textline_orientation", True),
                use_table_recognition=settings.get("enable_table", True),
                use_formula_recognition=settings.get("enable_formula", False),
                **self._device_kwargs(settings),
            )
        return self._engine

    def analyze(self, image: Any, page: int = 0) -> List[TextBlock]:
        """image: 图像 ndarray 或文件路径。返回有序块列表（默认按版面阅读顺序）。"""
        engine = self._ensure()
        if isinstance(image, np.ndarray):
            image = self._cap_size(image)
        result = engine.predict(image)
        blocks: List[TextBlock] = []
        for res in result or []:
            d = res.dict() if hasattr(res, "dict") else (res if isinstance(res, dict) else {})
            for item in d.get("parsing_res_list", []) or []:
                blk = self._to_block(item, page)
                if blk is not None:
                    blocks.append(blk)
        blocks = self._sort_reading(blocks)
        return blocks

    def _cap_size(self, img: np.ndarray) -> np.ndarray:
        """限制最长边，避免大图在 8GB 显卡上 OLED 溢出。"""
        max_side = int(self.cfg.get("ocr", "max_side_len", 1800))
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return img
        import cv2

        scale = max_side / float(longest)
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    def _to_block(self, item: Any, page: int) -> Optional[TextBlock]:
        if hasattr(item, "__dict__") or hasattr(item, "label"):
            label = getattr(item, "label", "") or ""
            content = getattr(item, "content", "") or ""
            bbox = getattr(item, "bbox", None) or [0, 0, 0, 0]
            index = getattr(item, "index", 0) or 0
        elif isinstance(item, dict):
            label = item.get("label", "") or ""
            content = item.get("content", "") or ""
            bbox = item.get("bbox", [0, 0, 0, 0]) or [0, 0, 0, 0]
            index = item.get("index", 0) or 0
        else:
            return None

        kind = _LABEL_MAP.get(str(label).strip().lower(), "text")
        # 表格 content 是 HTML，转 Markdown
        if kind == "table" and content.strip().lower().startswith("<"):
            content = _html_to_markdown(content)
        content = " ".join(content.split())
        if not content:
            return None
        return TextBlock(
            text=unicodedata.normalize("NFKC", content),
            kind=kind,
            page=page,
            bbox=[float(x) for x in bbox],
            order=int(index or 0),
            meta={"raw_label": str(label)},
        )

    def _sort_reading(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """按版面阅读顺序排序：（页码, 行y中心, 列x中心）。"""
        def key(b: TextBlock) -> tuple:
            bb = b.bbox or [0, 0, 0, 0]
            yc = (bb[1] + bb[3]) / 2.0
            xc = (bb[0] + bb[2]) / 2.0
            return (b.page, round(yc / 12.0), xc)

        ordered = sorted(blocks, key=key)
        for i, b in enumerate(ordered):
            b.order = i
        return ordered