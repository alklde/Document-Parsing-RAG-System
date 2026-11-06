#!/usr/bin/env python3
"""LLM 后端抽象：默认 llama.cpp server（OpenAI 兼容接口）。"""
from __future__ import annotations

from typing import List

import requests

from ..config import Config


class LlamaCppBackend:
    """llama.cpp server 的 OpenAI 兼容客户端。

    启动示例：
        llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf \
            --host 127.0.0.1 --port 8081 -ngl 99 -c 4096 -a qwen2.5-3b
    """

    def __init__(self, cfg: Config):
        s = cfg.section("llm")
        self.base_url = str(s.get("base_url", "http://127.0.0.1:8081")).rstrip("/")
        self.model = s.get("model", "qwen2.5-3b")
        self.api_key = s.get("api_key", "")
        self.temperature = float(s.get("temperature", 0.2))
        self.max_tokens = int(s.get("max_tokens", 512))
        self.timeout = int(s.get("timeout", 120))

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def available(self) -> bool:
        """llama.cpp server 加载完成时 /health 返回 200。"""
        for path in ("/health", "/v1/models"):
            try:
                r = requests.get(
                    f"{self.base_url}{path}",
                    headers=self._headers(),
                    timeout=5,
                )
                if r.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    def chat(self, messages: List[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def make_llm_backend(cfg: Config) -> LlamaCppBackend:
    """按 config.yaml 的 llm.backend 创建后端；当前仅实现 llama.cpp。"""
    backend = str(cfg.get("llm", "backend", "llamacpp")).lower()
    if backend in ("llamacpp", "llama.cpp"):
        return LlamaCppBackend(cfg)
    raise ValueError(f"不支持的 llm.backend: {backend}（当前仅支持 llamacpp）")


def build_prompt(question: str, chunks: list) -> List[dict]:
    """组装引用式 RAG prompt：编号资料 + 来源信息。"""
    refs = [
        f"[{i}] {chunk['text']}"
        for i, (chunk, _score) in enumerate(chunks, 1)
    ]
    context = "\n".join(refs)
    system = (
        "你是企业知识库文档问答助手。请依据提供的资料直接回答用户问题，"
        "资料通常已包含答案，请从中提取。不得编造资料外的内容。"
        "直接输出答案正文，不要输出'资料来源'、'资料编号'等引导词或前缀，"
        "如引用资料请在答案末尾用 [n] 标注来源序号。"
    )
    user = f"资料:\n{context}\n\n问题:{question}\n\n回答:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
