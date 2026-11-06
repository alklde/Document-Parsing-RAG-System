#!/usr/bin/env python3
"""解析层统一入口：按文件类型分发。"""
from __future__ import annotations

import os
from typing import List, Optional

from ..config import Config
from .models import ParsedDocument
from .ocr_backend import PaddleOcrBackend

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class DocumentParser:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._ocr: Optional[PaddleOcrBackend] = None

    @property
    def ocr(self) -> PaddleOcrBackend:
        if self._ocr is None:
            self._ocr = PaddleOcrBackend(self.cfg)
        return self._ocr

    def parse(self, path: str) -> ParsedDocument:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            from .pdf_parser import parse_pdf

            return parse_pdf(path, self.cfg, self.ocr)
        if ext == ".docx":
            from .docx_parser import parse_docx

            return parse_docx(path, self.cfg, self.ocr)
        if ext in _IMAGE_EXT:
            from .image_parser import parse_image

            return parse_image(path, self.cfg, self.ocr)
        raise ValueError(f"不支持的文件类型: {ext} (支持 pdf/docx/png/jpg)")

    def parse_batch(self, paths: List[str]) -> List[ParsedDocument]:
        out = []
        for p in paths:
            out.append(self.parse(p))
        return out

    def parse_dir(self, directory: str) -> List[ParsedDocument]:
        paths = [
            os.path.join(directory, f)
            for f in sorted(os.listdir(directory))
            if os.path.splitext(f)[1].lower() in {".pdf", ".png", ".jpg", ".jpeg", ".docx"}
        ]
        return self.parse_batch(paths)