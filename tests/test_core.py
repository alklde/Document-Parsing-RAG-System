#!/usr/bin/env python3
"""docagent 单元测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.document_parser.models import ParsedDocument, TextBlock
from src.document_parser.ocr_backend import _html_to_markdown
from src.document_parser.pdf_parser import _reflow_paragraphs, normalize_text
from src.retriever.chunker import build_chunks, doc_id_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_cfg() -> Config:
    return Config.load(os.path.join(ROOT, "config.yaml"), root=ROOT)


def test_normalize_nfkc():
    # SourceHanSans 的 ToUnicode 会产出 CJK 兼容字（如 U+F98E=年），NFKC 需归一化
    assert normalize_text("\uF98E") == "年"
    assert normalize_text("\uFA08") == "行"
    assert normalize_text("\uF98E\uF98E\uF967FA0800") is not None
    assert normalize_text("考勤 2016") == "考勤 2016"


def test_reflow_paragraphs_merge_wrapped_line():
    lines = ["每日工作时间为上", "午9:00至下午18:00,午休1小时。", "下一段开始。"]
    out = _reflow_paragraphs(lines)
    assert out[0] == "每日工作时间为上午9:00至下午18:00,午休1小时。"


def test_reflow_paragraphs_keep_table_rows():
    lines = ["各类假期天数如下表所示：", "假期类型\u0000天数", "年假\u0000每年10天"]
    out = _reflow_paragraphs(lines)
    assert "假期类型\u0000天数" in out
    assert "年假\u0000每年10天" in out


def test_markdown_table_html():
    html = "<html><body><table><tr><td>指标</td><td>金额</td></tr></table></body></html>"
    md = _html_to_markdown(html)
    assert "指标" in md and "金额" in md
    assert md.lstrip().startswith("|")


def _flat_doc(blocks):
    return ParsedDocument(source="t.md", doc_type="md", pages=1, blocks=blocks)


def test_chunker_table_and_text():
    doc = _flat_doc([
        TextBlock(text="这是一段足够长的正文标题示例文本用于测试", kind="title", page=0, bbox=(0, 0, 0, 0), order=0),
        TextBlock(text="| 指标 | 金额 |\n|---|---|\n| 营业收入 | 2.4亿元 |", kind="table", page=0, bbox=(0, 0, 0, 0), order=1),
        TextBlock(text="这是足够长的正文内容，用于验证文本块被放入索引", kind="text", page=0, bbox=(0, 0, 0, 0), order=2),
        TextBlock(text="插图占位", kind="figure", page=0, bbox=(0, 0, 0, 0), order=3),
    ])
    chunks = build_chunks([doc], get_cfg())
    kinds = {c["meta"]["kind"] for c in chunks}
    assert kinds == {"title", "table", "text"}
    assert chunks[0]["meta"]["doc_id"] == "t"


def test_chunker_skips_figure():
    doc = _flat_doc([TextBlock(text="插图", kind="figure", page=0, bbox=(0, 0, 0, 0), order=0)])
    assert build_chunks([doc], get_cfg()) == []


def test_doc_id_of():
    assert doc_id_of("/a/b/员工手册.pdf") == "员工手册"