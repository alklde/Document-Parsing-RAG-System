# 智能文档解析与多模态 RAG 问答系统

将 PDF（文字层/扫描件）、图片、Word 统一解析为结构化内容，经 bge-m3 向量化构建 Faiss 索引，由本地 llama.cpp + Qwen2.5-3B 生成带引用来源的回答。

## 特性

- **双通道 PDF 解析**：文字层直接抽取，纯图片/扫描页降级 OCR
- **版面分析与表格还原**：PP-StructureV3 输出标题、正文、表格等版面块，表格转 Markdown
- **中文语义增强**：NFKC 归一化、段落重排，修复 PDF 折行问题
- **检索与生成**：bge-m3 嵌入 + Faiss 召回，引用式 Prompt 生成答案
- **服务与评测**：FastAPI REST 接口，支持 Recall@k / MRR / 关键词命中评测

## 技术栈

Python · PyMuPDF · PaddleOCR · bge-m3 · Faiss · llama.cpp · FastAPI · pytest

## 架构

```
文档（PDF/图片/Word）
        │
        ▼
解析：文字层 / OCR 版面分析 → 结构化块
        │
        ▼
分块 → bge-m3 嵌入 → Faiss 索引
        │
问题 → 检索 top-k → 引用式 Prompt → llama.cpp → 答案 + 来源
```

## 目录

```
src/document_parser/   PDF、图片、Word 解析与 OCR
src/retriever/         分块、bge-m3 嵌入、Faiss 索引
src/rag/               Prompt 组装与 llama.cpp 客户端
src/eval/              检索与问答评测
src/api/               FastAPI 服务
src/cli.py             parse / index / ask / eval
scripts/               示例数据生成
tests/                 单元测试
```

## 快速开始

```bash
pip install -r requirements.txt
python scripts/make_samples.py

python -m src.cli parse data/source/员工手册.pdf -o /tmp/out.json
python -m src.cli index data/source --rebuild
python -m src.cli ask "试用期工资是多少？" --k 4

uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

运行前需准备：bge-m3 本地缓存、PaddleOCR 模型、llama.cpp server（默认 `:8081`）。配置可通过本地 `config.yaml` 覆盖（未随仓库提供）。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 服务状态 |
| POST | `/api/ask` | 文档问答 |
| POST | `/api/parse` | 解析单个文档 |
| POST | `/api/index` | 上传并增量建索引 |
| POST | `/api/index/rebuild` | 全量重建索引 |

## 测试

```bash
python -m pytest tests/ -q
# 11 passed
```

## 数据

示例文档由 `scripts/make_samples.py` 生成；评测需自行准备 `data/benchmark.json`。