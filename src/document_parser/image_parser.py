#!/usr/bin/env python3
"""图片解析：整图走 PaddleOCR + 版面/表格。"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from ..config import Config
from .models import ParsedDocument
from .ocr_backend import PaddleOcrBackend


def parse_image(path: str, cfg: Config, ocr: PaddleOcrBackend) -> ParsedDocument:
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blocks = ocr.analyze(img, page=0)
    return ParsedDocument(source=path, doc_type="image", pages=1, blocks=blocks, raw={"shape": list(img.shape)})