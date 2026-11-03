"""Build the presentation for the TMG301 paper.

Run with the bundled Python that has python-pptx:

    python build_deck.py
"""

from __future__ import annotations

import os

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "TMG301_What_Moves_the_Number.pptx")

NAVY = RGBColor(0x0B, 0x25, 0x45)
BLUE = RGBColor(0x1D, 0x4E, 0x89)
LIGHT = RGBColor(0xE8, 0xF0, 0xFB)
GREY = RGBColor(0x5B, 0x67, 0x70)
DARK = RGBColor(0x1B, 0x1B, 0x1B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xB3, 0x26, 0x1E)
GREEN = RGBColor(0x1E, 0x7B, 0x4F)
AMBER = RGBColor(0xB2, 0x6A, 0x00)

FONT = "Segoe UI"
FONT_BOLD = "Segoe UI Semibold"

W = Inches(13.333)
H = Inches(7.5)

presentation = Presentation()
presentation.slide_width = W
presentation.slide_height = H
BLANK = presentation.slide_layouts[6]


def slide() -> object:
    return presentation.slides.add_slide(BLANK)


def textbox(slide, left, top, width, height, text="", size=18, bold=False,
            color=DARK, align=PP_ALIGN.LEFT, font=FONT, spacing=1.0):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = spacing
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font if bold else FONT
    return box


def bullets(slide, items, left, top, width, height, size=18, spacing=1.15):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    first = True
    for item in items:
        level, text = item[0], item[1]
        bold = item[2] if len(item) > 2 else False
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.level = level
        paragraph.line_spacing = spacing
        paragraph.space_after = Pt(10 if level == 0 else 4)
        prefix = "•  " if level == 0 else "–  "
        if level == 2:
            prefix = ""
        run = paragraph.add_run()
        run.text = prefix + text
        run.font.size = Pt(size if level == 0 else size - 2)
        run.font.bold = bold
        run.font.color.rgb = NAVY if bold else DARK
        run.font.name = FONT
    return box


def header(slide, title, kicker=None, number=None):
    if kicker:
        textbox(slide, Inches(0.75), Inches(0.42), Inches(11.8), Inches(0.3),
                kicker.upper(), size=12, bold=True, color=BLUE)
        top = Inches(0.72)
    else:
        top = Inches(0.5)
    textbox(slide, Inches(0.75), top, Inches(11.9), Inches(0.75), title, size=28,
            bold=True, color=NAVY)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.45),
                                  Inches(1.4), Pt(4))
    line.fill.solid()
    line.fill.fore_color.rgb = BLUE
    line.line.fill.background()
    if number is not None:
        textbox(slide, Inches(12.2), Inches(7.0), Inches(0.8), Inches(0.3), str(number),
                size=11, color=GREY, align=PP_ALIGN.RIGHT)
    textbox(slide, Inches(0.75), Inches(7.0), Inches(9.0), Inches(0.3),
            "TMG301 · What Moves the Number?", size=11, color=GREY)


def callout(slide, text, left, top, width, height, color=BLUE, size=17):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = color
    shape.line.width = Pt(1.25)
    shape.adjustments[0] = 0.08
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = Inches(0.18)
    frame.margin_right = Inches(0.18)
    frame.margin_top = Inches(0.08)
    frame.margin_bottom = Inches(0.08)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.LEFT
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = NAVY
    run.font.name = FONT
    return shape


def table(slide, data, left, top, width, col_ratios, row_height=Inches(0.42),
          header_size=13, body_size=13, highlight_rows=(), highlight_color=None):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, left, top, width, row_height * rows)
    tbl = shape.table
    total = sum(col_ratios)
    for index, ratio in enumerate(col_ratios):
        tbl.columns[index].width = Emu(int(width * ratio / total))
    for row_index, row in enumerate(data):
        tbl.rows[row_index].height = row_height
        for col_index, value in enumerate(row):
            cell = tbl.cell(row_index, col_index)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            frame = cell.text_frame
            frame.word_wrap = True
            paragraph = frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if col_index == 0 else PP_ALIGN.CENTER
            run = paragraph.add_run()
            run.text = str(value)
            run.font.size = Pt(header_size if row_index == 0 else body_size)
            run.font.bold = row_index == 0 or row_index in highlight_rows
            run.font.color.rgb = WHITE if row_index == 0 else DARK
            run.font.name = FONT
            cell.fill.solid()
            if row_index == 0:
                cell.fill.fore_color.rgb = NAVY
            elif row_index in highlight_rows and highlight_color is not None:
                cell.fill.fore_color.rgb = highlight_color
            else:
                cell.fill.fore_color.rgb = WHITE if row_index % 2 else LIGHT
    return shape


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------- 1. title
s = slide()
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(2.6))
band.fill.solid()
band.fill.fore_color.rgb = NAVY
band.line.fill.background()
textbox(s, Inches(0.9), Inches(0.55), Inches(11.5), Inches(0.4),
        "BÁO CÁO BÀI BÁO KHOA HỌC · TMG301 TEXT MINING", size=13, bold=True,
        color=RGBColor(0x9F, 0xC0, 0xE8))
textbox(s, Inches(0.9), Inches(1.05), Inches(11.5), Inches(1.0),
        "What Moves the Number?", size=52, bold=True, color=WHITE)
textbox(s, Inches(0.9), Inches(2.95), Inches(11.4), Inches(1.0),
        "Evaluation and Encoder Choices Dominate Architecture in\nVietnamese Spam-Review Classification",
        size=22, color=BLUE, spacing=1.15)
textbox(s, Inches(0.9), Inches(4.5), Inches(11.4), Inches(1.2),
        "Phan Bá Song Toàn  ·  Trịnh Minh Bảo Trân  ·  Trần Quang Long", size=18, bold=True, color=NAVY)
textbox(s, Inches(0.9), Inches(5.05), Inches(11.4), Inches(0.5),
        "34 run · 3 seed · một protocol duy nhất · 24 tài liệu tham khảo", size=15, color=GREY)
textbox(s, Inches(0.9), Inches(6.55), Inches(11.4), Inches(0.4),
        "Câu hỏi của bài: trong bài toán này, lựa chọn nào thực sự làm thay đổi điểm số?", size=15,
        color=BLUE)
notes(s, "Chào thầy/cô và các bạn. Bài báo của nhóm đặt một câu hỏi rất cụ thể: trong bài toán "
         "phân loại review rác tiếng Việt, yếu tố nào thực sự làm thay đổi điểm macro-F1 — kiến trúc "
         "mô hình, hay cách xây dựng tập đánh giá? Nhóm chạy 34 cấu hình dưới cùng một protocol để "
         "trả lời câu hỏi đó bằng số liệu chứ không bằng cảm nhận.")

# ---------------------------------------------------------------- 2. problem
s = slide()
header(s, "Vì sao đánh giá rác trên sàn TMĐT tiếng Việt lại khó?", "Bối cảnh", 2)
bullets(s, [
    (0, "Review quyết định mua hàng → bị thao túng có động cơ (khen ảo, dìm hàng, spam)", True),
    (0, "Văn bản rất ngắn: trung vị 20 token, 95% dưới 68 token"),
    (0, "Teencode, viết tắt, xen emoji: “sp”, “ko”, “đc”, “nhaaaaaaaa”"),
    (0, "Copy-paste nguyên văn giữa các sản phẩm → mô hình có thể học thuộc văn mẫu"),
], Inches(0.75), Inches(1.8), Inches(6.6), Inches(3.6), size=17)
callout(s, "19.870 dòng dữ liệu\nnhưng chỉ 19.750 văn bản khác nhau", Inches(7.7), Inches(1.9),
        Inches(4.9), Inches(1.5), size=19)
callout(s, "→ Học thuộc mẫu vẫn được điểm cao\ntrên split ngẫu nhiên", Inches(7.7), Inches(3.7),
        Inches(4.9), Inches(1.5), color=AMBER, size=17)
notes(s, "Điểm cần nhấn: dữ liệu review tiếng Việt ngắn và nhiều teencode, nhưng vấn đề nghiêm trọng "
         "nhất là copy-paste. Có gần 20 nghìn dòng nhưng chỉ hơn 19.700 văn bản khác nhau — nghĩa là "
         "khoảng 120 dòng là bản sao. Nếu chia ngẫu nhiên, bản sao rơi vào cả train và test, mô hình "
         "chỉ cần nhớ mẫu là được điểm.")

# ---------------------------------------------------------------- 3. tasks
s = slide()
header(s, "Hai bài toán và một bộ dữ liệu", "Bài toán", 3)
bullets(s, [
    (0, "Task 1 — nhị phân: non-spam / spam", True),
    (0, "Task 2 — bốn lớp: non-spam + 3 loại spam", True),
    (0, "Dữ liệu: ViSpamReviews v2 (công khai trên Hugging Face)"),
    (0, "Split phát hành: 14.306 train · 1.590 dev · 3.974 test"),
    (0, "Test: 2.914 non-spam · 54 Spam-1 · 310 Spam-2 · 696 Spam-3"),
], Inches(0.75), Inches(1.85), Inches(6.4), Inches(3.4), size=17)
table(s, [
    ["Thành phần", "Số lượng"],
    ["Dòng (train/dev/test)", "14.306 / 1.590 / 3.974"],
    ["Từ vựng train", "26.651"],
    ["Độ dài TB", "26,2 token"],
    ["Nhãn hiếm nhất", "211 dòng (1,5%)"],
], Inches(7.5), Inches(1.9), Inches(5.0), [1.3, 1.0], body_size=14)
notes(s, "Bài toán 2 là bài toán chính. Lưu ý ngay con số 54: lớp Spam-1 chỉ có 54 dòng test, đây là "
         "nguồn bất định lớn nhất của toàn bộ kết quả và sẽ quay lại ở phần hạn chế.")

# ---------------------------------------------------------------- 4. labels
s = slide()
header(s, "Bốn nhãn — nhìn bằng ví dụ thật", "Dữ liệu", 4)
table(s, [
    ["Nhãn", "Ý nghĩa", "Ví dụ trong dữ liệu"],
    ["Non-spam", "Review có thông tin về sản phẩm",
     "“chất liệu vải tốt mát, chất lượng phù hợp với giá tiền”"],
    ["Spam-1", "Review giả: khen/dìm có chủ đích",
     "“Chất lượng sản phẩm tuyệt vời. Giao hàng cũng nhanh. Đóng gói OK. Nên mua nhé mọi người.”"],
    ["Spam-2", "Chỉ nói về shop, thương hiệu, vận chuyển",
     "“sản phẩm bao bì đóng gói khá ok còn chất lượng cần thời gian kiểm chứng thêm”"],
    ["Spam-3", "Không phải review: gõ loạn, lặp ký tự, emoji",
     "“Âjskskxjxndndjsjdjskssosojsjsisossksksjdhhsjsussusjsjsjsjs”"],
], Inches(0.75), Inches(1.75), Inches(11.85), [0.9, 2.5, 6.4], row_height=Inches(0.95),
    body_size=13)
notes(s, "Đây là slide để người nghe hiểu dữ liệu. Nhấn mạnh: Spam-2 và Spam-3 nhìn là biết, nhưng "
         "Spam-1 thì không — câu văn trôi chảy, tích cực, đúng ngữ pháp. Dấu hiệu duy nhất là văn mẫu "
         "lặp đi lặp lại.")

# ---------------------------------------------------------------- 5. why spam1
s = slide()
header(s, "Vì sao Spam-1 khó nhất", "Dữ liệu", 5)
bullets(s, [
    (0, "Câu trôi chảy, tích cực, đúng ngữ pháp — giống review thật", True),
    (0, "Dấu hiệu chính: văn mẫu lặp lại giữa nhiều sản phẩm"),
    (0, "Cụm “chất lượng sản phẩm tuyệt vời” xuất hiện trong 21,8% review Spam-1"),
    (0, "… nhưng chỉ 0,65% review non-spam trong train"),
], Inches(0.75), Inches(1.8), Inches(6.5), Inches(3.2), size=17)
callout(s, "Hệ quả: mô hình học đúng văn mẫu,\nnhưng bị phạt khi người thật\nvô tình dùng cùng cụm từ",
        Inches(7.6), Inches(1.9), Inches(5.0), Inches(1.9), color=AMBER, size=16)
callout(s, "9,4% false positive chứa cụm này\n(so với 0,39% ở ca đoán đúng)",
        Inches(7.6), Inches(4.1), Inches(5.0), Inches(1.5), size=16)
notes(s, "Đây là phát hiện định lượng đáng chú ý: mô hình không 'ngu', nó học đúng tín hiệu. Nhưng "
         "vì nhãn do người gán, một số review non-spam dùng đúng văn mẫu nên bị gán sai. Đó là nhiễu "
         "nhãn, và nó đặt trần cho điểm số.")

# ---------------------------------------------------------------- 6. questions
s = slide()
header(s, "Bốn câu hỏi nghiên cứu", "Mục tiêu", 6)
cards = [
    ("1", "Tái lập", "Kết quả công bố có lặp lại được trên bản công khai?"),
    ("2", "Giám sát", "Mục tiêu đa tác vụ có giúp bài toán 4 lớp?"),
    ("3", "Tổng quát", "Giữ được bao nhiêu khi sản phẩm trong test là mới?"),
    ("4", "Độ nhạy", "Lựa chọn nào thực sự làm thay đổi điểm, hơn cả nhiễu seed?"),
]
for index, (num, title, body) in enumerate(cards):
    left = Inches(0.75 + (index % 2) * 6.05)
    top = Inches(1.9 + (index // 2) * 2.35)
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.7), Inches(2.0))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.fill.background()
    box.adjustments[0] = 0.06
    textbox(s, left + Inches(0.3), top + Inches(0.22), Inches(0.8), Inches(0.6), num, size=30,
            bold=True, color=BLUE)
    textbox(s, left + Inches(1.15), top + Inches(0.25), Inches(4.3), Inches(0.5), title, size=20,
            bold=True, color=NAVY)
    textbox(s, left + Inches(1.15), top + Inches(0.85), Inches(4.3), Inches(0.9), body, size=15,
            color=GREY, spacing=1.1)
notes(s, "Bốn câu hỏi này chính là bốn phần của bài. Câu 4 là câu mới so với bản nháp đầu và là câu "
         "quan trọng nhất — cả bài là để trả lời nó.")

# ---------------------------------------------------------------- 7. protocol
s = slide()
header(s, "Một protocol duy nhất cho 34 run", "Phương pháp", 7)
table(s, [
    ["Thành phần", "Giá trị", "Vì sao quan trọng"],
    ["Encoder", "PhoBERT-base-v2 (135M)", "Giữ nguyên cho mọi run"],
    ["Độ dài tối đa", "100 token", "Đủ cho review ngắn"],
    ["Batch / epoch", "32 / 10, dừng sớm patience 3", "Dừng theo dev macro-F1"],
    ["Learning rate", "encoder 1e-5 · head 5e-5", "Hai nhóm tham số"],
    ["Seed", "42, 43, 44", "Đo được nhiễu seed"],
], Inches(0.75), Inches(1.8), Inches(11.8), [1.6, 3.2, 3.4], row_height=Inches(0.6),
    body_size=14)
callout(s, "Bản nháp đầu: baseline và multi-task chạy hai protocol khác nhau "
           "→ mức tăng 0,73 điểm là giả.", Inches(0.75), Inches(5.5), Inches(11.8), Inches(1.1),
        color=RED, size=16)
notes(s, "Điểm kỹ thuật quan trọng nhất: trước khi sửa, baseline dùng max_len 128 và một learning "
         "rate, còn multi-task dùng max_len 100 và hai learning rate. Như vậy chênh lệch 0,73 điểm có "
         "thể do protocol chứ không do kiến trúc. Toàn bộ bài này chạy lại dưới đúng một protocol.")

# ---------------------------------------------------------------- 8. imbalance
s = slide()
header(s, "Mất cân bằng: lớp hiếm quyết định metric", "Dữ liệu", 8)
rows = [("Non-spam", "10.512", "73,5%"), ("Spam-1", "211", "1,5%"),
        ("Spam-2", "1.103", "7,7%"), ("Spam-3", "2.480", "17,3%")]
top = Inches(1.9)
for index, (name, count, share) in enumerate(rows):
    y = top + Inches(index * 0.78)
    bar_width = Inches(6.6 * float(share.replace(",", ".").replace("%", "")) / 100 * 1.35)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.75), y,
                             max(bar_width, Inches(0.35)), Inches(0.55))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RED if name == "Spam-1" else BLUE
    bar.line.fill.background()
    textbox(s, Inches(0.8), y + Inches(0.08), Inches(3.0), Inches(0.4), name, size=16, bold=True,
            color=WHITE)
    textbox(s, Inches(7.6), y + Inches(0.06), Inches(2.2), Inches(0.4), f"{count} dòng", size=15,
            color=DARK)
    textbox(s, Inches(9.9), y + Inches(0.06), Inches(1.6), Inches(0.4), share, size=15, color=GREY)
callout(s, "Test set: chỉ 54 dòng Spam-1\n→ F1 của lớp này gần như không suy diễn được",
        Inches(7.6), Inches(5.2), Inches(4.9), Inches(1.3), color=RED, size=15)
notes(s, "Vì lớp hiếm nhất chỉ 1,5%, accuracy vô nghĩa: đoán tất cả là non-spam đã được 73%. Vì vậy "
         "bài dùng macro-F1 — mỗi lớp trọng số bằng nhau. Nhưng chính vì vậy, một lớp 54 dòng có thể "
         "làm dao động điểm rất mạnh.")

# ---------------------------------------------------------------- 9. leakage
s = slide()
header(s, "Lỗ hổng 1: rò rỉ theo sản phẩm", "Dữ liệu", 9)
bullets(s, [
    (0, "Split ngẫu nhiên theo review → cùng một sản phẩm nằm ở cả train và test", True),
    (0, "Từ vựng sản phẩm (tên model, shop, mô tả) rò từ train sang test"),
    (0, "Mô hình học thuộc ngữ cảnh sản phẩm thay vì học dấu hiệu spam"),
], Inches(0.75), Inches(1.8), Inches(6.3), Inches(2.6), size=17)
table(s, [
    ["Cặp split", "Nhóm sản phẩm trùng"],
    ["train ↔ dev", "996"],
    ["train ↔ test", "1.456"],
    ["dev ↔ test", "921"],
], Inches(7.5), Inches(1.85), Inches(4.9), [1.4, 1.0], body_size=15)
callout(s, "1.676 / 1.006 / 1.479 nhóm duy nhất →\nphần lớn sản phẩm xuất hiện ở nhiều split",
        Inches(0.75), Inches(4.6), Inches(11.8), Inches(1.1), color=AMBER, size=16)
notes(s, "Cách kiểm tra: gộp tên sản phẩm + category + mô tả thành một khoá, chuẩn hoá, rồi đếm khoá "
         "xuất hiện ở nhiều split. Kết quả: hơn 1.400 khoá nằm ở cả train và test. Đây là lý do split "
         "ngẫu nhiên thường lạc quan.")

# ---------------------------------------------------------------- 10. fixes
s = slide()
header(s, "Hai lỗ hổng, hai cách xử lý", "Dữ liệu", 10)
left_box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.75), Inches(1.8),
                              Inches(5.75), Inches(4.3))
left_box.fill.solid()
left_box.fill.fore_color.rgb = LIGHT
left_box.line.fill.background()
textbox(s, Inches(1.05), Inches(2.0), Inches(5.2), Inches(0.5), "① Tách theo sản phẩm", size=19,
        bold=True, color=NAVY)
bullets(s, [
    (0, "GroupShuffleSplit trên khoá sản phẩm"),
    (0, "Kết quả: 13.985 / 1.947 / 3.938"),
    (0, "Test: 2.961 / 49 / 284 / 644"),
    (0, "0 nhóm trùng giữa ba split"),
], Inches(1.05), Inches(2.65), Inches(5.2), Inches(2.8), size=15)
right_box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.85), Inches(1.8),
                               Inches(5.75), Inches(4.3))
right_box.fill.solid()
right_box.fill.fore_color.rgb = LIGHT
right_box.line.fill.background()
textbox(s, Inches(7.15), Inches(2.0), Inches(5.2), Inches(0.5), "② Soi trùng nguyên văn", size=19,
        bold=True, color=NAVY)
bullets(s, [
    (0, "38 dòng test trùng văn bản với train (split gốc)"),
    (0, "Sau khi tách sản phẩm: vẫn còn 30 dòng"),
    (0, "213 dòng nằm trong nhóm trùng lặp"),
    (0, "→ tách sản phẩm chưa đủ, cần cả mức văn bản"),
], Inches(7.15), Inches(2.65), Inches(5.2), Inches(2.8), size=15)
notes(s, "Hai lỗ hổng khác nhau. Tách theo sản phẩm xử lý lỗ hổng từ vựng sản phẩm. Nhưng vẫn còn "
         "lỗ hổng thứ hai: review spam được copy nguyên văn sang sản phẩm khác, nên 30 dòng test vẫn "
         "trùng train. Nhóm giữ nguyên các dòng này và báo cáo số lượng thay vì lặng lẽ xoá.")

# ---------------------------------------------------------------- 11. models
s = slide()
header(s, "Ba mô hình, một bộ mã hoá", "Phương pháp", 11)
boxes = [
    (0.9, 1.95, 2.5, 0.8, "Review", LIGHT, NAVY),
    (0.9, 3.25, 2.5, 0.8, "PhoBERT", RGBColor(0xDD, 0xE7, 0xF6), NAVY),
]
for left, top, width, height, label, fill, color in boxes:
    shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
                               Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = BLUE
    textbox(s, Inches(left + 0.1), Inches(top + 0.18), Inches(width - 0.2), Inches(0.4), label,
            size=16, bold=True, color=color, align=PP_ALIGN.CENTER)
arrow = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(1.98), Inches(2.8), Inches(0.35),
                           Inches(0.4))
arrow.fill.solid()
arrow.fill.fore_color.rgb = GREY
arrow.line.fill.background()
for index, (label, detail) in enumerate([
    ("Phẳng", "4 lớp trực tiếp"),
    ("Đa tác vụ", "nhị phân + loại spam"),
    ("Fusion", "+ mô tả sản phẩm"),
]):
    top = 4.4 + index * 0.85
    shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(top),
                               Inches(4.4), Inches(0.68))
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE
    shape.line.color.rgb = BLUE
    textbox(s, Inches(1.05), Inches(top + 0.13), Inches(1.7), Inches(0.4), label, size=15,
            bold=True, color=NAVY)
    textbox(s, Inches(2.6), Inches(top + 0.16), Inches(2.5), Inches(0.4), detail, size=13, color=GREY)
image = s.shapes.add_picture(os.path.join(ASSETS, "confusion.png"), Inches(6.1), Inches(1.8),
                             width=Inches(6.5))
image.crop_left = 0
textbox(s, Inches(6.1), Inches(1.55), Inches(6.5), Inches(0.3),
        "Minh hoạ kết quả: ma trận nhầm lẫn (slide 16)", size=12, color=GREY)
notes(s, "Mô hình phẳng: PhoBERT + một lớp linear 4 đầu ra. Mô hình đa tác vụ: cùng encoder, hai đầu "
         "ra — một nhị phân, một 3 lớp chỉ học trên review spam — rồi ghép bằng cascade. Fusion: mã "
         "hoá thêm mô tả sản phẩm. Điểm cần nói: cả ba dùng chung encoder nên so sánh là công bằng.")

# ---------------------------------------------------------------- 12. main result
s = slide()
header(s, "Kết quả chính: hai mô hình gần như bằng nhau", "Kết quả", 12)
table(s, [
    ["Mô hình", "Seed", "Accuracy", "Macro-F1"],
    ["Phẳng (flat)", "3", "90,89 ± 0,31", "76,10 ± 0,71"],
    ["Đa tác vụ (cascade)", "3", "90,87 ± 0,45", "76,02 ± 0,96"],
    ["Chênh lệch", "—", "−0,03", "−0,08"],
], Inches(0.75), Inches(1.85), Inches(11.8), [3.2, 1.0, 2.2, 2.2], row_height=Inches(0.62),
    body_size=15, highlight_rows=(3,), highlight_color=RGBColor(0xFF, 0xF4, 0xE5))
callout(s, "Khoảng tin cậy bootstrap: [−0,61 ; +3,94]  ·  p = 0,142  →  không phân biệt được với 0",
        Inches(0.75), Inches(4.6), Inches(11.8), Inches(1.0), size=17)
textbox(s, Inches(0.75), Inches(5.85), Inches(11.8), Inches(0.6),
        "Nếu chỉ báo cáo seed 42: 77,10 vs 75,35 — trông như một cải tiến. Seed 43 và 44 đảo lại.",
        size=15, color=GREY)
notes(s, "Đây là kết quả gây bất ngờ nhất: mô hình phức tạp hơn không hơn gì. Nhưng phải nói rất cẩn "
         "thận: không phải 'kém hơn', mà là 'không phân biệt được với 0'. Nếu chỉ chạy một seed, ta rất "
         "dễ kết luận sai — chính nhóm đã từng kết luận sai như vậy.")

# ---------------------------------------------------------------- 13. statistics
s = slide()
header(s, "Không khác nhau về điểm, nhưng sai khác nhau", "Kết quả", 13)
bullets(s, [
    (0, "Bootstrap 10.000 lần: chênh lệch −0,08 pp, CI chứa 0", True),
    (0, "McNemar trên 147 dòng bất đồng: p = 0,047"),
    (0, "Đa tác vụ sửa 86 câu, làm hỏng 61 câu → net +25"),
    (0, "Hai mô hình mắc lỗi ở những câu khác nhau, không mô hình nào trội hơn"),
], Inches(0.75), Inches(1.85), Inches(11.8), Inches(2.8), size=17)
table(s, [
    ["", "Phẳng", "Đa tác vụ"],
    ["Sửa được (đúng thêm)", "—", "86"],
    ["Làm hỏng (sai thêm)", "—", "61"],
    ["Tổng lỗi / 3.974", "368", "343"],
], Inches(0.75), Inches(4.5), Inches(6.4), [2.6, 1.4, 1.6], body_size=14)
callout(s, "Bài học: p-value của McNemar nói hai mô hình khác nhau,\nkhông nói mô hình nào tốt hơn",
        Inches(7.5), Inches(4.4), Inches(5.0), Inches(1.5), size=15)
notes(s, "Cần phân biệt hai loại kiểm định. Bootstrap trả lời: điểm trung bình có khác không — câu trả "
         "lời là không. McNemar trả lời: hai mô hình có mắc lỗi giống nhau không — câu trả lời là "
         "không, chúng sai ở các câu khác nhau. Đây là câu trả lời rất tốt cho câu hỏi phản biện.")

# ---------------------------------------------------------------- 14. effects
s = slide()
header(s, "Cái gì thực sự làm thay đổi điểm?", "Kết quả", 14)
chart_data = CategoryChartData()
# PowerPoint plots the first category at the bottom, so the list is built
# bottom-up: the seed-noise reference sits at the bottom, the largest mover on top.
chart_data.categories = ["Nhiễu seed (sd)", "Cửa sổ 128 token", "Đa tác vụ",
                         "Metadata fusion", "Bỏ trùng văn bản", "Backbone XLM-R",
                         "Một learning rate", "Tách theo sản phẩm"]
chart_data.add_series("Δ macro-F1", (0.96, 0.39, -0.08, -0.12, -1.38, -1.47, -1.54, -1.96))
chart = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.7),
                           Inches(7.6), Inches(4.9), chart_data).chart
chart.has_legend = False
chart.has_title = False
plot = chart.plots[0]
plot.gap_width = 45
plot.has_data_labels = True
labels = plot.data_labels
labels.font.size = Pt(12)
labels.font.bold = True
labels.number_format = '0.00'
labels.number_format_is_linked = False
colors = [GREY, GREEN, RED, RED, RED, RED, RED, RED]
for index, color in enumerate(colors):
    point = plot.series[0].points[index]
    point.format.fill.solid()
    point.format.fill.fore_color.rgb = color
chart.category_axis.tick_labels.font.size = Pt(12)
chart.value_axis.tick_labels.font.size = Pt(11)
chart.value_axis.has_major_gridlines = True
chart.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
table(s, [
    ["Nhóm", "Δ"],
    ["Kiến trúc (đa tác vụ)", "−0,08"],
    ["Metadata fusion", "−0,12"],
    ["Cửa sổ token", "+0,39"],
    ["Nhiễu seed", "0,7 – 1,0"],
], Inches(8.55), Inches(1.9), Inches(4.0), [2.4, 1.2], body_size=14)
callout(s, "Đổi cách chấm: 1,0–2,3 điểm\nĐổi kiến trúc: 0,1 điểm",
        Inches(8.55), Inches(4.4), Inches(4.0), Inches(1.3), size=16)
notes(s, "Đây là slide quan trọng nhất của bài. Đọc từ trên xuống: tách theo sản phẩm, learning rate, "
         "backbone và bỏ trùng văn bản đều 1,4 đến 2,0 điểm. Rồi đến metadata và kiến trúc, gần như bằng 0. "
         "Thanh xám ở dưới cùng là nhiễu seed 0,96. Nghĩa là hiệu ứng kiến trúc còn nhỏ hơn cả nhiễu.")

# ---------------------------------------------------------------- 15. reversal
s = slide()
header(s, "Đảo chiều khi đổi cách chấm", "Kết quả", 15)
table(s, [
    ["Cách đánh giá", "Phẳng", "Đa tác vụ", "Kết luận"],
    ["Split gốc, 3 seed", "76,10", "76,02", "Hòa"],
    ["Tách theo sản phẩm, 3 seed", "75,94", "73,98", "Phẳng hơn 1,96"],
    ["Bỏ trùng văn bản, 3 seed", "74,72", "77,02", "Đa tác vụ hơn 2,30"],
], Inches(0.75), Inches(1.85), Inches(11.8), [3.4, 1.6, 1.8, 2.6], row_height=Inches(0.68),
    body_size=15, highlight_rows=(1, 2, 3))
callout(s, "Ba cách đánh giá đều hợp lý — và chúng mâu thuẫn nhau.\n"
           "Không thể kết luận về kiến trúc khi chưa cố định cách đánh giá.",
        Inches(0.75), Inches(5.15), Inches(11.8), Inches(1.4), color=RED, size=17)
notes(s, "Đây là bằng chứng mạnh nhất cho luận điểm của bài. Cùng hai mô hình, ba cách chấm cho ba "
         "kết luận khác nhau. Không phải mô hình thay đổi, mà là thước đo thay đổi.")

# ---------------------------------------------------------------- 16. errors
s = slide()
header(s, "Lỗi nằm ở đâu: bỏ sót rác, không phải nhầm loại", "Phân tích lỗi", 16)
s.shapes.add_picture(os.path.join(ASSETS, "confusion.png"), Inches(0.6), Inches(1.7),
                     width=Inches(7.5))
table(s, [
    ["Loại lỗi", "Phẳng", "Đa tác vụ"],
    ["Rác → không rác", "226", "208"],
    ["Không rác → rác", "106", "106"],
    ["Nhầm loại rác", "36", "29"],
], Inches(8.35), Inches(1.9), Inches(4.3), [2.2, 1.0, 1.2], body_size=14)
callout(s, "Điểm nghẽn là quyết định nhị phân,\nkhông phải tầng phân loại",
        Inches(8.35), Inches(4.5), Inches(4.3), Inches(1.2), size=15)
notes(s, "Nhìn ma trận: hàng Spam-2 có 37% bị đoán thành non-spam. Ngược lại, nhầm giữa các loại rác "
         "chỉ 36 ca. Nghĩa là mô hình phát hiện rác chưa đủ tốt, chứ không phải phân loại rác sai. "
         "Hướng cải thiện nên tập trung vào quyết định nhị phân.")

# ---------------------------------------------------------------- 17. label noise
s = slide()
header(s, "Nhiễu nhãn: khi người thật dùng văn mẫu", "Phân tích lỗi", 17)
bullets(s, [
    (0, "9,4% false positive chứa cụm “chất lượng sản phẩm tuyệt vời”", True),
    (0, "Cụm này có trong 21,8% review Spam-1 của train"),
    (0, "Nhưng chỉ 0,65% review non-spam — và 0,39% ở các ca mô hình đoán đúng"),
    (0, "Không ca nào trong số đó là bản sao nguyên văn của train"),
], Inches(0.75), Inches(1.85), Inches(6.4), Inches(3.0), size=17)
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.75), Inches(5.0),
                         Inches(6.4), Inches(1.5))
box.fill.solid()
box.fill.fore_color.rgb = LIGHT
box.line.fill.background()
textbox(s, Inches(1.0), Inches(5.15), Inches(5.9), Inches(1.2),
        "“Chất lượng sản phẩm tuyệt vời. Đóng gói sản phẩm đẹp. Thời gian giao hàng nhanh”\n"
        "→ nhãn: Non-spam", size=13, color=NAVY, spacing=1.15)
callout(s, "Mô hình học đúng tín hiệu của dữ liệu,\nnhưng một phần bị phạt bởi nhãn",
        Inches(7.5), Inches(2.1), Inches(5.0), Inches(1.5), color=AMBER, size=16)
textbox(s, Inches(7.5), Inches(3.9), Inches(5.0), Inches(2.0),
        "Đây là suy luận gián tiếp: nhóm không gán nhãn lại toàn bộ dữ liệu, mà đo tỉ lệ trùng văn mẫu. "
        "Vì vậy con số 9,4% nên đọc như bằng chứng về hướng, không phải ước lượng chính xác.",
        size=13, color=GREY, spacing=1.15)
notes(s, "Ví dụ này rất dễ gây tranh luận, nên nói rõ: đây là bằng chứng gián tiếp. Nhóm không khẳng "
         "định 9,4% nhãn sai; chỉ nói rằng một phần false positive là do văn mẫu, và điều đó đặt trần "
         "cho điểm số.")

# ---------------------------------------------------------------- 18. negative results
s = slide()
header(s, "Kết quả âm: ba can thiệp hợp lý đều không giúp", "Kết quả", 18)
table(s, [
    ["Can thiệp", "Macro-F1", "So với chuẩn"],
    ["Đa tác vụ (chuẩn, seed 42)", "77,10", "—"],
    ["+ trọng số cân bằng lớp", "72,91", "−4,19"],
    ["Phẳng (chuẩn, seed 42)", "75,35", "—"],
    ["+ nối category & mô tả vào text", "76,66", "+1,31"],
    ["+ fusion mã hoá riêng mô tả", "75,98", "+0,63"],
], Inches(0.75), Inches(1.8), Inches(11.8), [4.6, 1.6, 2.0], row_height=Inches(0.55),
    body_size=14, highlight_rows=(2,))
callout(s, "Trọng số làm tệ hẳn: lớp 54 dòng bị phạt quá mức.\n"
           "Metadata không giúp kể cả khi không bị cắt cụt.",
        Inches(0.75), Inches(5.2), Inches(11.8), Inches(1.0), color=AMBER, size=16)
textbox(s, Inches(0.75), Inches(6.35), Inches(11.8), Inches(0.4),
        "Lưu ý: các dòng dùng seed 42 để so cùng một seed với chuẩn.", size=12, color=GREY)
notes(s, "Ba can thiệp đều là những ý tưởng hợp lý mà ai cũng nghĩ tới. Trọng số cân bằng lớp làm "
         "điểm tụt 4 điểm. Nối metadata vào text thì không hiệu quả vì 94,5% dòng bị cắt cụt. Nhóm "
         "làm thêm bản fusion mã hoá mô tả riêng — vẫn không hơn. Đây là kết quả âm có giá trị: nó "
         "tiết kiệm thời gian cho người sau.")

# ---------------------------------------------------------------- 19. conclusion
s = slide()
header(s, "Hạn chế, kết luận và những gì còn mở", "Kết luận", 19)
textbox(s, Inches(0.75), Inches(1.75), Inches(5.6), Inches(0.4), "Ba hạn chế đã biết", size=18,
        bold=True, color=RED)
bullets(s, [
    (0, "Spam-1 chỉ 54 dòng test → F1 bất định"),
    (0, "Split disjoint mới là một cách chia"),
    (0, "Fusion dùng chung encoder, chưa phải S-PhoBERT hai tầng"),
], Inches(0.75), Inches(2.25), Inches(5.6), Inches(2.4), size=14)
textbox(s, Inches(6.9), Inches(1.75), Inches(5.6), Inches(0.4), "Kết luận", size=18, bold=True,
        color=GREEN)
bullets(s, [
    (0, "Kiến trúc: ~0,1 điểm. Đánh giá: 1,0–2,3 điểm"),
    (0, "Backbone quan trọng hơn objective"),
    (0, "Lỗi chính là bỏ sót rác, không phải nhầm loại"),
    (0, "Kết quả âm cũng là kết quả"),
], Inches(6.9), Inches(2.25), Inches(5.6), Inches(2.4), size=14)
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.5), W, Inches(1.1))
band.fill.solid()
band.fill.fore_color.rgb = NAVY
band.line.fill.background()
textbox(s, Inches(0.75), Inches(5.72), Inches(11.9), Inches(0.6),
        "Câu hỏi không phải “mô hình nào tốt hơn”, mà là “đo bằng thước nào”.",
        size=22, bold=True, color=WHITE)
textbox(s, Inches(0.75), Inches(6.7), Inches(11.9), Inches(0.4),
        "Mã nguồn + 34 run + PDF: github.com/PhanBaSongToan/what-moves-the-number",
        size=13, color=GREY)
notes(s, "Kết lại: nếu chỉ mang về một câu, đó là câu ở dải xanh — câu hỏi đúng không phải mô hình nào "
         "tốt hơn, mà là đo bằng thước nào. Nhóm sẵn sàng nhận câu hỏi.")

presentation.save(OUT)
print("saved", OUT, "slides:", len(presentation.slides))
