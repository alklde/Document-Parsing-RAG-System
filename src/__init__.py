"""智能文档解析与多模态 RAG 问答系统。

在包加载早期设置 huggingface 环境变量，使 bge 嵌入模型可纯本地离线加载。
"""
from __future__ import annotations

import os

# 默认走 HF 镜像并离线读取本地缓存（用户可用环境变量覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")