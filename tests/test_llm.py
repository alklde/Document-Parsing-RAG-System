#!/usr/bin/env python3
"""llama.cpp 后端单元测试（不依赖真实服务）。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.rag.llm import LlamaCppBackend, build_prompt, make_llm_backend


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_llamacpp_chat_uses_openai_endpoint(monkeypatch):
    cfg = Config.load()
    backend = LlamaCppBackend(cfg)
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return FakeResponse(
            200,
            {"choices": [{"message": {"content": " 答案 [1] "}}]},
        )

    monkeypatch.setattr("src.rag.llm.requests.post", fake_post)
    out = backend.chat([{"role": "user", "content": "测试"}])

    assert out == "答案 [1]"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["json"]["model"] == "qwen2.5-3b"
    assert captured["json"]["max_tokens"] == 512
    assert captured["json"]["stream"] is False


def test_llamacpp_available_prefers_health(monkeypatch):
    cfg = Config.load()
    backend = LlamaCppBackend(cfg)
    calls = []

    def fake_get(url, headers, timeout):
        calls.append(url)
        return FakeResponse(200 if url.endswith("/health") else 503)

    monkeypatch.setattr("src.rag.llm.requests.get", fake_get)

    assert backend.available() is True
    assert calls[0].endswith("/health")


def test_make_llm_backend_rejects_unsupported_backend():
    cfg = Config.load()
    cfg.cfg["llm"]["backend"] = "legacy"

    with pytest.raises(ValueError, match="llamacpp"):
        make_llm_backend(cfg)


def test_build_prompt_numbers_sources():
    chunks = [
        ({"text": "年假 10 天", "meta": {}}, 0.9),
        ({"text": "试用期工资 80%", "meta": {}}, 0.8),
    ]
    messages = build_prompt("年假多少天？", chunks)

    assert messages[0]["role"] == "system"
    assert "[1] 年假 10 天" in messages[1]["content"]
    assert "[2] 试用期工资 80%" in messages[1]["content"]
    assert "问题:年假多少天？" in messages[1]["content"]
