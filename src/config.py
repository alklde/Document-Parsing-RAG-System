#!/usr/bin/env python3
"""全局配置：默认值 + YAML 覆盖。"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Optional

import yaml

_DEFAULTS: Dict[str, Any] = {
    # 解析
    "ocr": {
        "backend": "paddle",          # paddle / (预留 rapidocr)
        "lang": "ch",
        "device": "auto",             # auto | cuda | cpu
        "use_doc_orientation": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
        "enable_layout": True,        # PP-Structure 版面+表格解析
        "enable_table": True,
        "min_text_chars": 1,          # 过滤过短文本块（噪声）
    },
    # 分块
    "chunk": {
        "size": 400,
        "overlap": 80,
        "min_chunk_chars": 20,
    },
    # 嵌入
    "embed": {
        "model": "BAAI/bge-m3",       # 本地 HF 缓存
        "device": "auto",
        "batch_size": 8,
        "use_fp16": True,             # 与 Paddle OCR 挤压 8GB 显存时减半
        "hf_offline": True,           # 纯本地离线读取（模型需已缓存）
        "query_instruction": "为这个句子生成表示以用于检索相关文章：",
    },
    # 向量库
    "faiss": {
        "index_dir": "data/indexed",
        "index_type": "flat_ip",       # 归一化后等价余弦
    },
    # 生成（llama.cpp server，OpenAI 兼容接口）
    "llm": {
        "backend": "llamacpp",
        "model": "qwen2.5-3b",         # llama-server 的 --alias
        "base_url": "http://127.0.0.1:8081",
        "api_key": "",                 # llama-server 启用 --api-key 时填写
        "temperature": 0.2,
        "max_tokens": 512,
        "timeout": 120,
        "top_k": 4,                    # 召回条数
    },
    # 数据
    "data": {
        "source_dir": "data/source",
    },
}


@dataclasses.dataclass
class Config:
    cfg: Dict[str, Any]
    root: str = ""

    @classmethod
    def load(cls, path: Optional[str] = None, root: str = "") -> "Config":
        merged = _merge(_DEFAULTS, {})
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                file_cfg = yaml.safe_load(f) or {}
            merged = _merge(merged, file_cfg)
        return cls(merged, root)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.cfg.get(section, {}).get(key, default)

    def section(self, name: str) -> Dict[str, Any]:
        return self.cfg.get(name, {})

    @property
    def index_dir(self) -> str:
        d = self.get("faiss", "index_dir")
        return d if os.path.isabs(d) else os.path.join(self.root, d)

    @property
    def source_dir(self) -> str:
        d = self.get("data", "source_dir")
        return d if os.path.isabs(d) else os.path.join(self.root, d)


def _merge(base: Dict, extra: Dict) -> Dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out