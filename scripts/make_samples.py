#!/usr/bin/env python3
"""生成示例数据集（三种形态）：
  1. data/source/员工手册.pdf        — 文字层 PDF（表格+章节）
  2. data/source/合同扫描件.pdf      — 无文字层的“扫描”PDF（验证 OCR 路径）
  3. data/source/季度经营报告.png    — 图片文档（标题+段落+表格）
"""
from __future__ import annotations

import os

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "source")
os.makedirs(OUT, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Light.otf",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/wenquanyi/wqy-zenhei.ttc",
]
def _find_font() -> str:
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            return f
    raise FileNotFoundError("未找到中文字体")


def _pdf_wrap(text: str, size: float, width_pt: float) -> list[str]:
    """按近似字符宽度（中文=size，ASCII≈size*0.5）贪心换行，避免 insert_text 溢出丢弃。"""
    def w(ch: str) -> float:
        if ord(ch) < 128:
            return size * 0.5
        return size

    lines, cur, cur_w = [], "", 0.0
    for ch in text:
        cw = w(ch)
        if cur_w + cw > width_pt and cur:
            lines.append(cur)
            cur, cur_w = ch, cw
        else:
            cur += ch
            cur_w += cw
    if cur:
        lines.append(cur)
    return lines


def _pdf_lines(doc: pymupdf.Document, x: float, y: float, text: str, size: float = 11,
               width_pt: float = 547.0, gap: float = 16.0) -> None:
    """在给定起始点逐行写入（内部先换行），排版稳定不漏字。"""
    page = doc[-1]
    for ln in _pdf_wrap(text, size, width_pt):
        page.insert_text((x, y), ln, fontsize=size, fontfile=_find_font(), fontname="cjk")
        y += gap


def make_handbook():
    font = _find_font()
    doc = pymupdf.open()
    # --- 封面
    page = doc.new_page()
    page.insert_text((36, 60), "星辰科技员工手册", fontsize=22, fontfile=font, fontname="cjk")
    page.insert_text((36, 90), "——考勤、薪酬与发展制度（2026 版）", fontsize=12, fontfile=font, fontname="cjk")
    # --- 页2
    doc.new_page()
    _pdf_lines(doc, 36, 40, "第一章 总则", size=16)
    _pdf_lines(doc, 36, 64, "本手册适用于星辰科技有限公司全体员工。公司实行每周五天工作制，工作日为周一至周五，每日工作时间为上午9:00至下午18:00，午休1小时。加班需提前申请并经部门主管审批。")
    _pdf_lines(doc, 36, 96, "第二章 考勤与假期", size=16)
    _pdf_lines(doc, 36, 121, "员工休假须提前填写请假申请单，经直属主管批准后方可休假。各类假期天数如下表所示：")
    trows = [
        ("假期类型", "员工有权享受的天数"),
        ("年假", "每年10天"),
        ("婚假", "15天"),
        ("产假", "158天"),
        ("陪产假", "10天"),
        ("病假", "按实际需休"),
        ("事假", "每年不超过20天"),
    ]
    ty = 152
    for c1, c2 in trows:
        page2 = doc[-1]
        page2.insert_text((36, ty), f"{c1}\t{c2}", fontsize=11, fontfile=font, fontname="cjk")
        ty += 20
    _pdf_lines(doc, 36, ty + 10, "年假按入职年限递增，工作满8年可增至15天。法定节假日依据国家规定执行。")
    # --- 页3
    doc.new_page()
    _pdf_lines(doc, 36, 40, "第三章 薪酬福利", size=16)
    _pdf_lines(doc, 36, 64, "公司实行十三薪制度，每年12月发放年终绩效奖金。社会保险与住房公积金按法律法规缴纳。试用期工资为转正工资的80%。")
    _pdf_lines(doc, 36, 100, "第四章 培训与发展", size=16)
    _pdf_lines(doc, 36, 124, "公司每年投入研发培训经费预算的5%用于员工技能提升。新员工入职后需完成两周的岗前培训。")
    _pdf_lines(doc, 36, 160, "第五章 离职管理", size=16)
    _pdf_lines(doc, 36, 184, "员工提出离职需提前30天书面通知公司；试用期内提前3天通知即可。离职后1年内不得从事与公司有竞争关系的工作，违反者需承担相应责任。")
    # --- 页4
    doc.new_page()
    _pdf_lines(doc, 36, 40, "附录：联系方式", size=14)
    _pdf_lines(doc, 36, 64, "人力资源部：hr@xingchen.com，电话 010-88886666。")

    out = os.path.join(OUT, "员工手册.pdf")
    doc.save(out)
    print("生成:", out)


def _pil_wrap(dr, font, text, max_w):
    """按像素宽度对中英文混合文本做贪心换行。"""
    lines, cur = [], ""
    for ch in text:
        if dr.textlength(cur + ch, font=font) <= max_w:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def make_scanned_contract():
    """无文字层 PDF：中文通过 PIL 渲染为图像再插入，模拟扫描件。"""
    from PIL import Image, ImageDraw, ImageFont

    font = _find_font()
    f_span = ImageFont.truetype(font, 30)
    f_head = ImageFont.truetype(font, 54)
    f_table = ImageFont.truetype(font, 28)

    W, H = 1500, 1754  # A4@150dpi，加宽避免溢出行
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    dr.text((W // 2 - 240, 60), "房屋租赁合同", font=f_head, fill="black")
    dr.text((80, 160), "合同编号：XC-ZL-2026-018", font=f_span, fill="black")

    lines = [
        "出租方（甲方）：北京星辰物业管理有限公司",
        "承租方（乙方）：张三",
        "",
        "第一条 租赁标的：甲方将位于北京市海淀区知春路88号二层写字楼（建筑面积120平方米）出租给乙方。",
        "第二条 租赁期限：自 2026 年 1 月 1 日起至 2028 年 12 月 31 日止，租期三年。租赁期满后同等条件下乙方享有优先续租权。",
        "第三条 租金与支付：月租金为人民币 12000 元，按季度支付。乙方须支付押金人民币 24000 元（押二付一）。",
        "第四条 违约责任：任意一方违约须向对方支付相当于 2 个月租金的违约金。乙方逾期支付租金超过 15 日，甲方有权解除合同。",
        "第五条 维修责任：房屋主体结构维修由甲方负责，日常损耗维修由乙方负责。",
        "乙方应付费用的支付方式：银行转账。",
    ]
    y = 280
    for ln in lines:
        if not ln:
            y += 40
            continue
        for sub in _pil_wrap(dr, f_span, ln, W - 160):
            dr.text((80, y), sub, font=f_span, fill="black")
            y += 55
        y += 15

    # 费用明细表
    y += 30
    dr.text((80, y), "费用明细表", font=f_span, fill="black")
    y += 60
    table = [
        ("费用项目", "金额", "周期"),
        ("租金", "12000 元", "月"),
        ("押金", "24000 元", "一次性"),
        ("物业费", "800 元", "月"),
    ]
    col_w = [380, 340, 260]
    for row in table:
        x = 80
        for j, cell in enumerate(row):
            dr.rectangle([x, y, x + col_w[j], y + 50], outline="black", width=2)
            dr.text((x + 14, y + 8), cell, font=f_table, fill="black")
            x += col_w[j]
        y += 50

    tmp_img = os.path.join(OUT, "_contract_tmp.png")
    img.save(tmp_img)

    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    page.insert_image(page.rect, filename=tmp_img)
    out = os.path.join(OUT, "合同扫描件.pdf")
    doc.save(out)  # 无文字层，纯图片
    os.remove(tmp_img)
    print("生成:", out, "(纯图片PDF，将走OCR路径)")


def make_report_image():
    from PIL import Image, ImageDraw, ImageFont

    font = _find_font()
    f_head = ImageFont.truetype(font, 46)
    f_span = ImageFont.truetype(font, 28)
    f_table = ImageFont.truetype(font, 26)

    W, H = 1600, 1600
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    dr.text((W // 2 - 360, 60), "星辰科技 2026 年第二季度经营报告", font=f_head, fill="black")
    dr.text((80, 160), "报告日期：2026-07-15", font=f_span, fill="black")

    paras = [
        "2026 年第二季度，公司整体经营情况良好，营业收入达到 2.4 亿元，同比增长 18%，环比增长 6%。",
        "报告期内净利润 3800 万元，同比增长 12%。研发投入 6800 万元，同比增长 25%，占比进一步提升，体现了公司对技术创新的持续重视。",
        "在产品线方面，AI 应用与云服务业务增长最为迅速，收入占比首次超过 40%。海外市场收入 5200 万元，同比增长 30%。",
    ]
    y = 260
    for p in paras:
        for sub in _pil_wrap(dr, f_span, p, W - 160):
            dr.text((80, y), sub, font=f_span, fill="black")
            y += 64
        y += 40

    dr.text((80, y + 10), "主要财务指标", font=f_span, fill="black")
    y += 90
    table = [
        ("指标", "金额", "同比"),
        ("营业收入", "2.4 亿元", "+18%"),
        ("净利润", "3800 万元", "+12%"),
        ("研发投入", "6800 万元", "+25%"),
        ("海外收入", "5200 万元", "+30%"),
    ]
    col_w = [340, 380, 340]
    for row in table:
        x = 80
        for j, cell in enumerate(row):
            dr.rectangle([x, y, x + col_w[j], y + 58], outline="black", width=2)
            dr.text((x + 18, y + 12), cell, font=f_table, fill="black")
            x += col_w[j]
        y += 58

    out = os.path.join(OUT, "季度经营报告.png")
    img.save(out)
    print("生成:", out)


if __name__ == "__main__":
    make_handbook()
    make_scanned_contract()
    make_report_image()
    print("全部完成:", OUT)