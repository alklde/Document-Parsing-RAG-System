#!/usr/bin/env python3
"""本地嵌入：BGE 系模型（bge-m3 默认）。"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import Config


class BgeEmbedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = cfg.get("embed", "model", "BAAI/bge-m3")
        self.query_instr = cfg.get("embed", "query_instruction", "")
        self.batch_size = int(cfg.get("embed", "batch_size", 8))
        self._model = None

    def _load(self):
        if self._model is None:
            import os

            # 默认走镜像（可被环境变量覆盖）
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            # 纯本地离线读取，避免联网探测导致加载失败
            if self.cfg.get("embed", "hf_offline", True):
                os.environ["HF_HUB_OFFLINE"] = "1"

            from sentence_transformers import SentenceTransformer

            device = self.cfg.get("embed", "device", "auto")
            dev = None if device == "auto" else device
            self._model = SentenceTransformer(self.model_name, device=dev)

            use_fp16 = bool(self.cfg.get("embed", "use_fp16", True))
            # 与 Paddle OCR 共挤 8GB 显存，用 fp16 减半
            if use_fp16 and dev != "cpu" and self._device_is_cuda():
                self._model.half()
                # fp32 权重的残留会滞留在 torch 显存缓存池，必须显式释放
                import torch

                torch.cuda.empty_cache()
        return self._model

    def _device_is_cuda(self) -> bool:
        import torch

        return torch.cuda.is_available()

    def _to_float32(self, arr) -> np.ndarray:
        a = np.asarray(arr)
        return a.astype("float32") if a.dtype != "float32" else a

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        q = f"{self.query_instr}{text}" if self.query_instr else text
        vec = model.encode(
            [q],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vec, dtype="float32")[0]

    @property
    def dim(self) -> int:
        model = self._load()
        return model.get_sentence_embedding_dimension()


class DummyEmbedder(BgeEmbedder):
    """占位：小规模/函数测试使用随机向量，验证全链路。"""

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        out = rng.normal(size=(len(texts), 128)).astype("float32")
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out

    def embed_query(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(0)
        v = rng.normal(size=128).astype("float32")
        return v / np.linalg.norm(v)

    @property
    def dim(self) -> int:
        return 128