"""AI 服务 HTTP 层：把文档解析 + RAG 问答封装为 REST API，供 Java OA 后端调用。

启动（agent 仓库根目录）：
    python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000

端点：
    GET  /api/health          服务状态（索引条数、llama.cpp 可达性）
    POST /api/ask             {question, k?} -> {answer, sources, model, elapsed_ms}
    POST /api/parse           multipart file -> 结构化版面块 JSON
    POST /api/index           multipart file -> 解析并入索引（增量）
    POST /api/index/rebuild   从 data/source 全量重建索引
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import Config

cfg = Config.load(os.path.join(ROOT, "config.yaml"), root=ROOT)
UPLOAD_DIR = os.path.join(ROOT, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".docx"}

# Paddle/PyTorch 非线程安全，重操作串行化（OA 场景并发量足够）
_heavy_lock = threading.Lock()

app = FastAPI(title="DocAgent AI Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=4, ge=1, le=20)


class AskSource(BaseModel):
    source: str
    page: int
    kind: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[AskSource]
    model: str
    elapsed_ms: int


# ---------------------------------------------------------------- 内部工具
def _get_store():
    from src.retriever.embedder import BgeEmbedder
    from src.retriever.faiss_store import FaissStore

    return FaissStore(cfg, BgeEmbedder(cfg))


def _save_upload(file: UploadFile, content: bytes) -> str:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"不支持的文件类型 {ext}，允许: {sorted(ALLOWED_EXT)}")
    name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename or 'upload')}"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _free_gpu():
    """解析(OCR)完成后释放显存，给 bge 嵌入腾空间。"""
    import gc

    gc.collect()
    try:
        import paddle

        paddle.device.cuda.empty_cache()
    except Exception:
        pass
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------- 端点
@app.get("/api/health")
def health() -> Dict[str, Any]:
    store_ok = os.path.exists(os.path.join(cfg.index_dir, "index.faiss"))
    index_count = 0
    if store_ok:
        import json

        with open(os.path.join(cfg.index_dir, "meta.json"), "r", encoding="utf-8") as f:
            index_count = len(json.load(f))
    from src.rag.llm import LlamaCppBackend

    llamacpp_ok = LlamaCppBackend(cfg).available()
    return {
        "status": "ok",
        "index_count": index_count,
        "llamacpp": llamacpp_ok,
        "llm_model": cfg.get("llm", "model"),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> Dict[str, Any]:
    if not os.path.exists(os.path.join(cfg.index_dir, "index.faiss")):
        raise HTTPException(400, "知识库索引为空，请先上传文档建立索引")
    from src.rag.generator import RagGenerator

    t0 = time.perf_counter()
    with _heavy_lock:
        store = _get_store()
        hits = store.search(req.question, k=req.k)
        if not hits:
            raise HTTPException(404, "未检索到相关内容，请换个问法或先补充知识库文档")
        result = RagGenerator(cfg).answer(req.question, hits)
    result["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    return result


@app.post("/api/parse")
def parse(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = file.file.read()
    if not content:
        raise HTTPException(400, "空文件")
    path = _save_upload(file, content)
    from src.document_parser.pipeline import DocumentParser

    with _heavy_lock:
        dp = DocumentParser(cfg)
        parsed = dp.parse(path)
        _free_gpu()
    js = parsed.to_json()
    js["saved_path"] = os.path.relpath(path, ROOT)
    return js


@app.post("/api/index")
def index_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = file.file.read()
    if not content:
        raise HTTPException(400, "空文件")
    path = _save_upload(file, content)
    from src.document_parser.pipeline import DocumentParser
    from src.retriever.chunker import build_chunks

    with _heavy_lock:
        dp = DocumentParser(cfg)
        parsed = dp.parse(path)
        _free_gpu()
        chunks = build_chunks([parsed], cfg)
        store = _get_store()
        store.add(chunks)
        total = store.count()
    return {
        "file": os.path.basename(file.filename or ""),
        "blocks": len(parsed.blocks),
        "chunks": len(chunks),
        "index_total": total,
        "saved_path": os.path.relpath(path, ROOT),
    }


@app.post("/api/index/rebuild")
def index_rebuild() -> Dict[str, Any]:
    from src.document_parser.pipeline import DocumentParser
    from src.retriever.chunker import build_chunks

    with _heavy_lock:
        dp = DocumentParser(cfg)
        docs = dp.parse_dir(cfg.source_dir)
        _free_gpu()
        chunks = build_chunks(docs, cfg)
        store = _get_store()
        store.rebuild(chunks)
        total = store.count()
    return {"docs": len(docs), "chunks": len(chunks), "index_total": total}
