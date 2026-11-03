"""Build the Vietnamese reading version of the paper as a DOCX."""

from __future__ import annotations

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "BAN_TIENG_VIET.docx")

NAVY = RGBColor(0x0B, 0x25, 0x45)
GREY = RGBColor(0x55, 0x5F, 0x66)

doc = Document()

section = doc.sections[0]
section.page_width = Cm(21.0)
section.page_height = Cm(29.7)
section.left_margin = Cm(2.4)
section.right_margin = Cm(2.2)
section.top_margin = Cm(2.2)
section.bottom_margin = Cm(2.2)

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15


def title(text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(17)
    run.font.color.rgb = NAVY
    paragraph.paragraph_format.space_after = Pt(4)


def subtitle(text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(11)
    run.font.color.rgb = GREY
    paragraph.paragraph_format.space_after = Pt(12)


def h1(text: str) -> None:
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(13)
    run.font.color.rgb = NAVY
    paragraph.paragraph_format.space_before = Pt(14)
    paragraph.paragraph_format.space_after = Pt(6)


def h2(text: str) -> None:
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = True
    run.italic = True
    run.font.size = Pt(11.5)
    paragraph.paragraph_format.space_before = Pt(10)
    paragraph.paragraph_format.space_after = Pt(4)


def p(text: str, justify: bool = True) -> None:
    paragraph = doc.add_paragraph(text)
    if justify:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Cm(0.6)


def plain(text: str) -> None:
    paragraph = doc.add_paragraph(text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def caption(text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(9.5)
    paragraph.paragraph_format.space_before = Pt(8)
    paragraph.paragraph_format.space_after = Pt(3)


def table(rows: list[list[str]], widths: list[float] | None = None) -> None:
    shape = doc.add_table(rows=len(rows), cols=len(rows[0]))
    shape.style = "Light Grid Accent 1"
    shape.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            cell = shape.cell(row_index, col_index)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(str(value))
            run.font.size = Pt(9)
            run.bold = row_index == 0
            if widths:
                cell.width = Cm(widths[col_index])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ------------------------------------------------------------------ title
title("Điều gì làm thay đổi con số?")
subtitle("Lựa chọn đánh giá và lựa chọn encoder chi phối kiến trúc\n"
         "trong phân loại review rác tiếng Việt")
plain("Phan Ba Song Toan, Trinh Minh Bao Tran, Tran Quang Long — Da Nang 550000, Vietnam")
note = doc.add_paragraph()
note.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = note.add_run("Bản dịch tiếng Việt để đọc. Bản chính thức nộp kèm là bản tiếng Anh (paper_ieee_vispam.pdf).")
run.italic = True
run.font.size = Pt(9.5)
run.font.color.rgb = GREY

h1("Tóm tắt")
p("Review thương mại điện tử tiếng Việt được gán một trong bốn nhãn: không rác, review giả, review chỉ nói về "
  "thương hiệu, hoặc không phải review; một bộ phân loại PhoBERT đã đạt mức giữa 70 ở macro-F1 cho bài bốn lớp "
  "này. Các bài trong lĩnh vực thường báo một con số trên một cách chia duy nhất rồi quy phần cải thiện cho kiến "
  "trúc. Chúng tôi kiểm tra trực tiếp quy kết đó. Dưới một giao thức huấn luyện duy nhất, chúng tôi chạy 53 cấu "
  "hình và đo mức mỗi lựa chọn thiết kế dịch chuyển macro-F1: mục tiêu đa tác vụ phân cấp, fusion metadata mã hoá "
  "mô tả sản phẩm ở một lượt riêng, backbone, cửa sổ token, lịch learning rate, và hai lựa chọn đánh giá — bỏ "
  "văn bản huấn luyện trùng nguyên văn với tập test, và giữ sản phẩm ngoài tập huấn luyện. Các lựa chọn kiến trúc "
  "chỉ dịch chuyển metric tối đa một phần mười điểm. Các lựa chọn đánh giá dịch chuyển 1,4 đến 2,3 điểm và đảo "
  "thứ hạng hai mô hình; đổi backbone mất 1,5 điểm. Chỉ riêng nhiễu seed đã là 0,7 đến 1,0. Chúng tôi cũng báo "
  "cáo phân rã lỗi cho thấy bỏ sót spam — chứ không phải nhầm loại spam — chi phối phần lỗi còn lại, cùng một "
  "ước lượng định lượng về nhiễu nhãn ở các review không rác mang văn mẫu.")
p("Từ khoá: phát hiện review rác, xử lý ngôn ngữ tiếng Việt, PhoBERT, học đa tác vụ, đánh giá tách sản phẩm, "
  "tái lập kết quả.")

h1("I. Giới thiệu")
p("Review quyết định việc khám phá sản phẩm trên các sàn thương mại điện tử, và cũng chính động cơ khiến chúng có "
  "giá trị biến chúng thành mục tiêu: người bán có thể đăng văn bản quảng cáo, đối thủ có thể đăng văn bản bôi "
  "nhọ, và cả hai đều có thể tự động hoá. Các sàn Việt Nam thêm một khó khăn nữa. Văn bản người dùng rất ngắn, "
  "phi chính thức, nhiều viết tắt và teencode, xen emoji, và thường được copy nguyên văn giữa các sản phẩm, nên "
  "một bộ phân loại có thể đạt điểm khá bằng cách học thuộc văn mẫu thay vì đánh giá nội dung.")
p("Bộ dữ liệu ViSpamReviews và phiên bản mở rộng có metadata hình thức hoá bài toán thành hai nhiệm vụ. Task 1 "
  "là nhị phân: không rác so với rác. Task 2 là bốn lớp: không rác cộng ba loại rác. Công trình trước báo cáo "
  "PhoBERT là backbone mạnh nhất khi chỉ dùng văn bản, và fusion metadata cải thiện thêm. Hai tính chất của cách "
  "đánh giá đó giới hạn mức độ khái quát của con số. Thứ nhất, cách chia là ngẫu nhiên trên review, nên cùng một "
  "sản phẩm có thể xuất hiện ở cả train và test; từ vựng riêng của sản phẩm và văn mẫu được copy khi đó thổi "
  "phồng điểm. Thứ hai, bốn lớp không độc lập: một review không thể vừa không rác vừa là một loại rác, nên một "
  "bộ phân loại phẳng bốn lớp bỏ qua cấu trúc mà chính tập nhãn đã mã hoá.")
p("Bài này cố ý là một nghiên cứu nhỏ, kiểm toán được, chứ không phải một suất trên bảng xếp hạng. Chúng tôi đặt "
  "bốn câu hỏi và trả lời mỗi câu bằng các artifact có thể chạy lại:")
for item in [
    "Tái lập. Kết quả PhoBERT đã công bố có lấy lại được từ bản công khai không, và hai snapshot dữ liệu khác nhau ở đâu?",
    "Giám sát. Một mục tiêu hai đầu phân cấp — huấn luyện một bộ phát hiện nhị phân và một bộ phân loại loại rác "
    "trên cùng encoder — có cải thiện metric bốn lớp khi cả hai mô hình dùng chung một giao thức và được đánh giá "
    "trên nhiều seed không?",
    "Khái quát hoá. Bao nhiêu độ chính xác phụ thuộc vào việc thấy cùng sản phẩm trong lúc huấn luyện, và trùng lặp "
    "ở mức văn bản có còn là rò rỉ sau khi đã tách theo sản phẩm?",
    "Độ nhạy. Trong tất cả các lựa chọn đó, cái nào thực sự làm thay đổi metric, và các hiệu ứng có lớn hơn nhiễu seed?",
]:
    paragraph = doc.add_paragraph(item, style="List Number")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p("Đóng góp của chúng tôi gồm: (i) một bảng effect-size đặt mọi lựa chọn thiết kế lên cùng một thang, lấy nhiễu "
  "seed làm thước đo, và cho thấy các lựa chọn đánh giá lớn hơn các lựa chọn kiến trúc một bậc độ lớn; (ii) một "
  "so sánh đơn giao thức có tài liệu, trong đó khác biệt duy nhất giữa hai mô hình là mục tiêu giám sát, với ba "
  "seed, khoảng tin cậy bootstrap theo cặp và kiểm định McNemar chính xác; (iii) một cách chia tách sản phẩm kèm "
  "kiểm toán rò rỉ tường minh, mở rộng bằng kiểm toán trùng nguyên văn mà cách đánh giá gốc không báo cáo; (iv) "
  "một phân rã định lượng phần lỗi còn lại, tách phát hiện rác khỏi phân loại rác, cộng một ước lượng phần nhiễu "
  "nhãn ở các review không rác mang văn mẫu; và (v) các kết quả âm cho giám sát có trọng số lớp, nối metadata thô "
  "và một baseline fusion tránh được cắt cụt. Chúng tôi cũng nói rõ những con số thăm dò trước đây không so sánh "
  "được với các con số ở đây, vì chúng được tạo ra trước khi giao thức được khoá.")

h1("II. Công trình liên quan")
h2("II.1. Opinion spam và phát hiện review rác")
p("Jindal và Liu đưa opinion spam thành bài toán phân loại văn bản và chỉ ra rằng review trùng lặp và gần trùng "
  "lặp là tín hiệu mạnh. Ott và cộng sự chỉ ra rằng người làm thuê có thể được yêu cầu viết review đánh lừa và các "
  "bộ phân loại n-gram tách chúng gần như hoàn hảo, một kết quả mà công trình sau quy một phần cho cách xây dựng "
  "lớp dương. Mukherjee và cộng sự chuyển từ từng review sang nhóm người viết và sang việc một bộ lọc thương mại "
  "thực sự loại bỏ những gì; Rayana và Akoglu ghép xếp hạng với phân loại. Phần kiểm toán của chúng tôi tiếp nối "
  "mạch đó.")
h2("II.2. Review rác tiếng Việt")
p("ViSpamReviews cung cấp khoảng hai mươi nghìn review có gán nhãn từ các sàn Việt Nam với quy trình gán nhãn "
  "chặt, và định nghĩa hai bài toán được dùng ở đây. Công trình nối tiếp về tích hợp metadata thêm category và mô "
  "tả sản phẩm qua sentence embedding và báo cáo rằng PhoBERT với embedding S-PhoBERT dẫn xuất từ mô tả đạt điểm "
  "tốt nhất. Công việc của chúng tôi khác ở hai điểm: chúng tôi đánh giá một mục tiêu phân cấp thay vì một đầu vào "
  "giàu hơn, và chúng tôi thêm cách chia tách sản phẩm cùng kiểm toán trùng lặp. Chúng tôi coi kết quả metadata là "
  "một baseline cần kiểm lại, không phải một khẳng định cần mở rộng.")
h2("II.3. Encoder pretrained tiếng Việt")
p("PhoBERT áp dụng công thức RoBERTa cho tiếng Việt với mô hình base 135 triệu tham số và là backbone mặc định "
  "cho phân loại câu tiếng Việt. BERT, biến thể tối ưu hoá mạnh hơn RoBERTa, và XLM-R đa ngữ là các lựa chọn thay "
  "thế thường gặp; thực hành fine-tune cho phân loại văn bản, gồm pooling và learning rate theo tầng, được tổng "
  "hợp trong tài liệu tương ứng. Chúng tôi giữ nguyên backbone và chỉ thay đổi cách giám sát, nên so sánh không "
  "phụ thuộc vào lựa chọn pretraining.")
h2("II.4. Giám sát đa tác vụ và phân cấp")
p("Caruana hình thức hoá học đa tác vụ như chuyển giao quy nạp qua một biểu diễn chia sẻ; Ruder khảo sát không "
  "gian thiết kế, và MT-DNN cho thấy các tác vụ phụ giúp hiểu ngôn ngữ ở quy mô lớn. Tập nhãn phân cấp được xử lý "
  "bằng bộ phân loại phẳng, bằng một chuỗi bộ phân loại cục bộ, hoặc bằng một mô hình toàn cục áp đặt phân cấp; "
  "tài liệu khảo sát so sánh ba cách. Cascade hấp dẫn khi quyết định cha dễ hơn quyết định con, đúng với trường "
  "hợp này: phát hiện rác dễ hơn nhiều so với gọi tên loại rác, và đầu loại chỉ nhận giám sát trên các ví dụ rác. "
  "Chúng tôi kiểm tra xem cấu trúc đó có trả công không, thay vì giả định.")
h2("II.5. Mất cân bằng và đánh giá")
p("Bài bốn lớp mất cân bằng nặng, loại hiếm nhất chiếm khoảng 1,5% số dòng huấn luyện. Re-weighting, focal loss, "
  "logit adjustment và resampling là các biện pháp chuẩn; tài liệu khảo sát hành vi của chúng trong học sâu. Vì "
  "một seed trên vài nghìn dòng test mang phương sai đáng kể, chúng tôi báo ba seed và dùng bootstrap theo cặp "
  "cùng kiểm định McNemar, cả hai đều phù hợp để so hai bộ phân loại trên cùng một tập test.")

h1("III. Dữ liệu và kiểm toán")
h2("III.1. Nhiệm vụ và nhãn")
p("Bảng I nhắc lại tập nhãn kèm một ví dụ huấn luyện mỗi lớp. Định nghĩa theo tài liệu bộ dữ liệu: No-Spam là "
  "review có thông tin về sản phẩm; Spam-1 là review giả bôi nhọ hoặc quảng bá nhân tạo; Spam-2 chỉ bàn về thương "
  "hiệu, người bán hoặc dịch vụ giao hàng và không nói gì về chất lượng sản phẩm; Spam-3 không phải review, "
  "thường là gõ loạn phím, lặp ký tự hoặc emoji. Các ví dụ cho thấy vì sao bài toán khó ngay cả với người: Spam-1 "
  "thường trôi chảy, đúng ngữ pháp và tích cực, chỉ phân biệt được chủ yếu nhờ văn mẫu.")
caption("Bảng I. Định nghĩa nhãn và một ví dụ huấn luyện nguyên văn mỗi lớp.")
table([
    ["Nhãn", "Ý nghĩa", "Ví dụ"],
    ["No-Spam", "Review có thông tin sản phẩm", "“chất liệu vải tốt mát nói chung là chất lượng phù hợp với giá tiền”"],
    ["Spam-1", "Review giả: khen hoặc dìm có chủ đích", "“Chất lượng sản phẩm tuyệt vời. Giao hàng cũng nhanh. Đóng gói sản phẩm OK. Nên mua nhé mọi người.”"],
    ["Spam-2", "Chỉ nói thương hiệu, người bán, giao hàng", "“sản phẩm bao bì đóng gói khá ok còn chất lượng cần thời gian kiểm chứng thêm”"],
    ["Spam-3", "Không phải review: gõ loạn, lặp, emoji", "“Âjskskxjxndndjsjdjskssosojsjsisossksksjdhhsjsussusjsjsjsjs”"],
], widths=[2.0, 5.0, 9.0])

h2("III.2. Bản công khai và bản đã công bố")
p("Chúng tôi dùng bản công khai của ViSpamReviews v2. Bảng II so bản này với cách chia mô tả trong bài công bố. "
  "Số dòng khác ở mọi split (14.306 / 1.590 / 3.974 so với 14.284 / 1.589 / 3.963), và số dòng từng lớp trên "
  "test cũng khác (2.914 / 54 / 310 / 696 so với 2.911 / 54 / 309 / 689). Hai bản không phải cùng một snapshot, "
  "nên việc tái lập phải được đọc là tái lập gần đúng, không phải chạy lại chính xác. Chúng tôi không so độ dài "
  "trung bình giữa hai bản, vì thống kê công bố tính sau tách từ tiếng Việt còn của chúng tôi đếm token theo "
  "khoảng trắng; hai đơn vị không quy đổi được cho nhau.")
caption("Bảng II. Bản công khai dùng ở đây so với cách chia trong bài công bố.")
table([
    ["Nguồn", "Split", "Số dòng", "Non-spam", "Spam-1", "Spam-2", "Spam-3", "Từ vựng", "Token TB"],
    ["Công bố", "Train", "14.284", "10.490", "211", "1.103", "2.480", "17.196", "17,38"],
    ["", "Dev", "1.589", "1.182", "17", "120", "270", "4.414", "16,92"],
    ["", "Test", "3.963", "2.911", "54", "309", "689", "8.144", "17,37"],
    ["Bản công khai", "Train", "14.306", "10.512", "211", "1.103", "2.480", "26.651", "26,19"],
    ["", "Dev", "1.590", "1.183", "17", "120", "270", "6.514", "25,84"],
    ["", "Test", "3.974", "2.914", "54", "310", "696", "12.189", "26,19"],
])
p("Phân bố nhãn lệch mạnh: 73,5% review huấn luyện là không rác, loại rác hiếm nhất chiếm 1,5% số dòng huấn "
  "luyện và 54 dòng test. Với phân bố này, accuracy bị lớp đa số chi phối, nên macro-F1 là metric chính xuyên "
  "suốt bài.")
h2("III.3. Rò rỉ ở mức sản phẩm")
p("Hai review của cùng một sản phẩm chia sẻ từ vựng không liên quan gì đến tính rác: tên model, kích cỡ, văn bản "
  "soạn sẵn của người bán. Một cách chia ngẫu nhiên theo review do đó làm rò rỉ danh tính sản phẩm. Để đo, chúng "
  "tôi chuẩn hoá tên sản phẩm, category và mô tả của từng dòng (NFKC, chữ thường, gộp khoảng trắng) rồi coi chuỗi "
  "ghép là một khoá sản phẩm. Bản công khai có 1.676 khoá phân biệt ở train, 1.006 ở dev và 1.479 ở test, và các "
  "khoá trùng nhau nhiều: 996 giữa train và dev, 1.456 giữa train và test, 921 giữa dev và test. Mọi điểm trên "
  "split ngẫu nhiên đều lạc quan so với triển khai thực tế, nơi sản phẩm mới đến liên tục.")
h2("III.4. Cách chia tách sản phẩm")
p("Cho thí nghiệm độ bền, chúng tôi dựng lại cách chia ở mức sản phẩm. Toàn bộ dòng được gộp và nhóm theo khoá "
  "chuẩn hoá; GroupShuffleSplit thứ nhất giữ 20% số nhóm làm test, lần thứ hai chia phần còn lại thành train và "
  "dev với 12,5% giữ lại. Kết quả có 13.985 / 1.947 / 3.938 dòng với support từng lớp 10.255/194/1.108/2.428, "
  "1.393/39/141/374 và 2.961/49/284/644. Theo cách dựng, không khoá sản phẩm nào được chia sẻ giữa các split, và "
  "chúng tôi kiểm tra tường minh điều đó; việc gán dòng được phát hành dưới dạng file chỉ mục để có thể dựng lại "
  "cách chia từ bản công khai mà không phải gửi ba bản dữ liệu. Vì cùng một lần gộp đã xoá ranh giới train/test "
  "ban đầu, tập test tách sản phẩm là một mẫu mới, không phải bản lọc của tập cũ, nên chúng tôi luôn so các mô "
  "hình trong cùng một split thay vì đọc chênh lệch giữa hai split như một thuộc tính của mô hình.")
h2("III.5. Trùng nguyên văn còn sót sau khi tách sản phẩm")
p("Nhóm theo sản phẩm loại một lỗ rò nhưng không loại lỗ kia. Một số văn mẫu rác được copy sang sản phẩm khác, "
  "nên một review có thể xuất hiện nguyên văn ở cả train và test dù không sản phẩm nào được chia sẻ. So sánh sau "
  "khi gộp khoảng trắng, cắt và bỏ phân biệt hoa thường, chúng tôi thấy 38 dòng test (35 văn bản khác nhau) của "
  "split gốc cũng xuất hiện trong train; trên toàn bộ bản công khai, 213 dòng thuộc các nhóm trùng lặp và 19.870 "
  "dòng chỉ chứa 19.750 văn bản khác nhau. Ở split tách sản phẩm, 30 dòng test vẫn trùng nguyên văn với train. "
  "Các dòng trùng không rải đều: trong 114 dòng huấn luyện bị trùng, 51 là không phải review (Spam-3), so với 11 "
  "review giả và 10 review chỉ nói thương hiệu — đúng như kỳ vọng nếu văn mẫu được copy để làm dày chứ không để "
  "thuyết phục. Ảnh hưởng lên metric tổng hợp là nhỏ, nhưng nó quan trọng với khẳng định rằng split tách sản phẩm "
  "đo khái quát hoá tới sản phẩm mới: nó đo khái quát hoá tới sản phẩm mới, không phải tới văn bản mới. Chúng tôi "
  "giữ các dòng này và báo cáo con số thay vì lặng lẽ xoá.")

h1("IV. Phương pháp")
h2("IV.1. Giao thức khoá")
p("Mọi con số ở Mục V đến từ một script và một giao thức duy nhất (Bảng III). Một run được xác định bởi nhiệm "
  "vụ, mục tiêu giám sát, biến thể đầu vào và seed; không gì khác thay đổi. Chúng tôi báo đầy đủ giao thức vì các "
  "run thăm dò trước đây, làm trước khi giao thức được cố định, không so sánh được với các con số dưới đây.")
caption("Bảng III. Giao thức huấn luyện khoá dùng chung cho mọi run được báo cáo.")
table([
    ["Thành phần", "Giá trị"],
    ["Encoder", "vinai/phobert-base-v2 (135M)"],
    ["Độ dài tối đa", "100 token"],
    ["Padding", "động, theo batch, bội số của 8"],
    ["Batch size", "32"],
    ["Epoch", "10, dừng sớm theo macro-F1 trên dev"],
    ["Patience", "3"],
    ["Optimizer", "AdamW, weight decay 0,01"],
    ["LR encoder / head", "1×10⁻⁵ / 5×10⁻⁵"],
    ["Gradient clipping", "1,0"],
    ["Precision", "bfloat16 autocast"],
    ["Loss", "cross-entropy (nhị phân, loại)"],
    ["Seed", "42, 43, 44"],
    ["Phần cứng", "NVIDIA RTX 5050 Laptop, 8 GB"],
], widths=[5.0, 9.0])
p("Giao thức khác các run thăm dò ở ba lựa chọn kỹ thuật: padding động theo batch, batch lớn hơn (32 thay vì 16) "
  "và bfloat16 autocast. Padding và precision thay đổi tính toán chứ không đổi mục tiêu; batch lớn hơn giảm số "
  "bước tối ưu trong một số epoch cố định, thêm một lý do để không trộn hai thế hệ run. Một run mất khoảng 11 "
  "phút trên phần cứng trên.")
h2("IV.2. Baseline đơn tác vụ")
p("Baseline là mô hình phân loại chuỗi PhoBERT chuẩn: một lớp linear trên biểu diễn của token đầu. Với Task 2, "
  "đầu ra có bốn chiều và huấn luyện bằng cross-entropy bốn lớp; với Task 1 là hai chiều. Encoder và bộ phân loại "
  "tạo hai nhóm tham số với learning rate ở Bảng III. Mô hình này cố ý đơn giản: nó tách ảnh hưởng của mục tiêu "
  "giám sát khỏi thay đổi backbone, dữ liệu hay tiền xử lý.")
h2("IV.3. Mô hình đa tác vụ")
p("Mô hình đề xuất chia sẻ một encoder giữa đầu nhị phân và đầu ba lớp loại rác, cả hai đều dropout–linear với "
  "dropout 0,2. Gọi b ∈ {0,1} là chỉ số rác và t ∈ {1,2,3} là loại rác. Đầu loại chỉ được huấn luyện trên các ví "
  "dụ rác:")
plain("L = L_bin(b) + 1[y_bin = 1] · L_type(t),        (1)")
p("trong đó 1[·] là hàm chỉ báo. Khi suy luận, quyết định nhị phân điều khiển quyết định loại: dự đoán không rác "
  "thành lớp 0, ngược lại lấy argmax của đầu loại cộng một. Cho phân tích độ tin cậy, chúng tôi cũng giữ tích các "
  "marginal:")
plain("p(c = 0) = p(b = 0),   p(c = k) = p(b = 1) · p(t = k − 1),        (2)")
p("đó là điều một cascade giả định khi coi hai đầu là độc lập. Công thức (2) chỉ dùng để xem độ tin cậy và dựng "
  "phân rã lỗi cổng ở Mục V.7; dự đoán được báo cáo đến từ cascade cứng.")
h2("IV.4. Baseline fusion metadata")
p("Để kiểm tra metadata có giúp khi không bị cắt cụt, chúng tôi thêm một mô hình fusion: review được mã hoá như "
  "trước, mô tả sản phẩm được mã hoá ở lượt forward thứ hai qua cùng encoder, hai biểu diễn được nối và một lớp "
  "linear bốn chiều áp lên kết quả. Không đầu vào nào bị cắt cụt. Đây là tái cài đặt trung thành ý tưởng fusion "
  "sentence-embedding của bài công bố với một sai lệch: chúng tôi dùng chung một encoder thay vì fine-tune một "
  "sentence encoder riêng ở tầng đầu, nên pipeline hai tầng đã công bố không được tái lập. Dòng này tồn tại để tách "
  "“metadata không giúp” khỏi “metadata bị cắt cụt”, điều mà ablation nối chuỗi không làm được.")
h2("IV.5. Đánh giá")
p("Accuracy được báo để so với công trình trước; macro-F1 là metric chính vì nó cho bốn lớp trọng số bằng nhau. "
  "Trừ khi nói khác, mọi giá trị là trung bình và độ lệch chuẩn trên seed 42, 43, 44, chọn theo macro-F1 trên dev "
  "và đánh giá một lần trên test. Với cặp chính, chúng tôi cũng báo bootstrap theo cặp trên các dòng test (10.000 "
  "lần lấy mẫu lại của chênh lệch macro-F1) và kiểm định McNemar chính xác trên các dự đoán bất đồng, vì cả hai "
  "đều so hai mô hình trên cùng đầu vào. Precision, recall, F1 từng lớp và ma trận nhầm lẫn đầy đủ được lưu cùng "
  "dự đoán.")
h2("IV.6. Ablation dưới giao thức khoá")
p("Hai can thiệp được chạy lại dưới giao thức khoá để so được với cặp chính: loss loại có trọng số nghịch tần số "
  "bên trong mục tiêu đa tác vụ, và nối metadata, trong đó category và mô tả sản phẩm được ghép vào văn bản review "
  "cho mô hình đơn tác vụ. Biến thể metadata tái lập định dạng đầu vào của các thí nghiệm fusion thăm dò nhưng "
  "không phải fusion sentence-embedding của bài công bố; do đó chúng tôi coi nó là ablation của một biến thể đầu "
  "vào, không phải tái cài đặt phương pháp đã công bố.")

h1("V. Kết quả")
h2("V.1. Tái lập")
p("Bảng IV đặt tham chiếu đã công bố cạnh các run giao thức khoá của chúng tôi và một tham chiếu từ vựng. Mô hình "
  "đơn tác vụ PhoBERT đạt 76,10 macro-F1 ở Task 2 và 89,46 ở Task 1 (accuracy 91,73), so với 71,56 và 85,58 trong "
  "bài công bố. Khoảng cách là dương nhưng không được đọc là cải thiện so với phương pháp gốc: hai thí nghiệm "
  "dùng snapshot khác nhau, bản công khai có thêm 11 dòng test, và giao thức của chúng tôi khác về batch size, "
  "padding và độ chính xác số. Điều bảng này xác lập là một baseline nội bộ được đo trong cùng điều kiện với mọi "
  "dòng khác trong bài. Dòng TF-IDF cho thấy mốc kỳ vọng: unigram và bigram với SVM tuyến tính chỉ đạt 66,52 "
  "macro-F1 ở Task 2.")
caption("Bảng IV. Tham chiếu công bố so với các run của chúng tôi và một baseline từ vựng.")
table([
    ["Cấu hình", "Task 1 (Acc / F1)", "Task 2 (Acc / F1)"],
    ["Bài gốc công bố", "89,23 / 85,58", "87,79 / 71,56"],
    ["Nhóm tái lập", "91,73 / 89,46", "90,89 / 76,10"],
    ["Tham chiếu TF-IDF", "84,63 / 78,80", "84,40 / 66,52"],
])
h2("V.2. Giám sát đa tác vụ")
p("Bảng V báo cáo so sánh chính trên ba seed. Mô hình đa tác vụ đạt 76,02 macro-F1 so với 76,10 của baseline, "
  "nên mục tiêu phân cấp đáng giá −0,08 điểm — nhỏ hơn độ lệch giữa các seed của cả hai mô hình và nằm gọn trong "
  "khoảng tin cậy. Hai mô hình thực sự bất đồng trên từng dòng: bootstrap theo cặp ở seed 42 cho khoảng "
  "[−0,61; +3,94] và kiểm định McNemar chính xác trên 147 dòng bất đồng cho p = 0,047, nghĩa là hai mô hình mắc "
  "lỗi khác nhau chứ không phải một mô hình mắc ít lỗi hơn. Đó là tóm tắt trung thực của so sánh này, và nó ngược "
  "với điều một lần chạy một seed của cùng đoạn mã gợi ý.")
caption("Bảng V. So sánh chính trên split chính thức, trung bình ba seed.")
table([
    ["Mô hình", "Accuracy", "Macro-F1"],
    ["Đơn tác vụ (baseline)", "90,89 ± 0,31", "76,10 ± 0,71"],
    ["Đa tác vụ (cascade)", "90,87 ± 0,45", "76,02 ± 0,96"],
    ["Chênh lệch", "−0,03", "−0,08"],
])
h2("V.3. Task 1")
p("Cùng mô hình đa tác vụ có sẵn một đầu nhị phân, nên điểm Task 1 có được miễn phí. Bảng VI so nó với một bộ "
  "phân loại nhị phân chuyên dụng. Mô hình chuyên dụng đạt 89,46 macro-F1 so với 89,15 của đầu nhị phân: cách "
  "nhau một phần ba điểm, nằm gọn trong độ lệch seed của cả hai. Huấn luyện đồng thời do đó không làm hỏng cũng "
  "không cải thiện quyết định nhị phân mà nó phụ thuộc vào, trong khi mô hình đa tác vụ trả lời cả hai câu hỏi "
  "trong một lượt forward.")
caption("Bảng VI. Task 1: bộ phân loại nhị phân chuyên dụng so với đầu nhị phân của mô hình đa tác vụ.")
table([
    ["Mô hình", "Accuracy", "Macro-F1"],
    ["Nhị phân chuyên dụng (3 seed)", "91,73 ± 0,43", "89,46 ± 0,53"],
    ["Đầu nhị phân của mô hình đa tác vụ", "91,68 ± 0,37", "89,15 ± 0,37"],
])
h2("V.4. Đánh giá tách sản phẩm")
p("Bảng VII lặp lại so sánh trên split tách sản phẩm với cùng ba seed. Trong split này, bộ phân loại phẳng đạt "
  "75,94 macro-F1 so với 73,98 của cascade, chênh 1,96 điểm nghiêng về mô hình phẳng, và hướng chênh lệch giữ "
  "nguyên ở hai trong ba seed. Hai cách đọc. Thứ nhất, không mô hình nào sụp khi mọi sản phẩm trong test đều mới, "
  "nên điểm trên split chính thức không phải sản phẩm phụ của trùng lặp sản phẩm. Thứ hai, lợi thế cascade thể "
  "hiện trên một seed của split chính thức không sống sót khi sản phẩm mới; nếu có gì thì phân cấp là mô hình kém "
  "bền hơn, ngược với lập luận transfer thường dùng cho biểu diễn chia sẻ. Chúng tôi báo chênh lệch như mô tả chứ "
  "không kiểm định, vì ba seed trên một mẫu test mới không đủ xác lập chênh lệch hai điểm.")
caption("Bảng VII. Đánh giá tách sản phẩm, cùng ba seed.")
table([
    ["Mô hình", "Accuracy", "Macro-F1"],
    ["Đơn tác vụ", "90,99 ± 0,42", "75,94 ± 1,02"],
    ["Đa tác vụ (cascade)", "90,67 ± 0,20", "73,98 ± 1,51"],
    ["Chênh lệch", "−0,31", "−1,96"],
])
h2("V.5. Ablation dưới giao thức khoá")
p("Bảng VIII báo cáo hai can thiệp. Đánh trọng số nghịch tần số cho loss loại rõ ràng có hại: macro-F1 giảm còn "
  "72,91 so với 77,10 của mô hình đa tác vụ không trọng số ở cùng seed, và mất mát tập trung ở hai loại rác lớn "
  "hơn — đúng những lớp mà trọng số làm giảm mẫu. Nối category và mô tả là câu chuyện khác: mô hình đạt 76,66 so "
  "với 75,35 của baseline chỉ văn bản ở cùng seed, chênh lệch mà một seed không phân giải được. So sánh đó cũng "
  "yếu hơn vẻ ngoài, vì mô tả sản phẩm trong bản công khai có độ dài trung vị 315 token theo khoảng trắng; với "
  "comment đặt trước, 94,5% số dòng vượt cửa sổ 100 token và phần sống sót là comment cộng một mảnh mô tả bị cắt. "
  "Run do đó kiểm tra nối chuỗi trong điều kiện cắt cụt, không phải tích hợp metadata như vậy. Baseline fusion ở "
  "Mục IV.4 loại bỏ nhiễu này bằng cách mã hoá mô tả riêng, và được bàn cùng các đối chứng khác.")
caption("Bảng VIII. Hai ablation dưới giao thức khoá (seed 42).")
table([
    ["Biến thể", "Accuracy", "Macro-F1"],
    ["Đa tác vụ (tham chiếu)", "91,37", "77,10"],
    ["+ loss loại có trọng số nghịch tần số", "89,43", "72,91"],
    ["Đơn tác vụ (tham chiếu)", "90,74", "75,35"],
    ["+ nối category và mô tả vào văn bản", "90,66", "76,66"],
])
h2("V.6. Đối chứng và độ nhạy")
p("Bảng IX tập hợp các đối chứng, mỗi cái chạy lại so với tham chiếu của nó dưới giao thức khoá. Các đối chứng "
  "rò rỉ đến trước: bỏ 47 dòng huấn luyện trùng với một dòng test đưa mô hình phẳng về 74,72 và cascade về 77,02; "
  "bỏ 211 dòng gần trùng thay vào đó cho 73,66 và 74,46. Cả hai mô hình mất nhiều hơn khi bỏ gần trùng, điều này "
  "nói rằng split ngẫu nhiên hào phóng vì văn mẫu được copy ở dạng hơi khác nhau, không chỉ nguyên văn.")
p("Hai đối chứng phía đầu vào chỉ về hai hướng ngược nhau. Chuẩn hoá teencode và viết tắt chat (ko → không, sp → "
  "sản phẩm) là can thiệp duy nhất có giúp: 77,56 cho mô hình phẳng so với 76,10. Metadata thì không giúp: fusion "
  "đạt 75,98 với encoder dùng chung và 75,95 với encoder riêng đóng băng, so với cùng 76,10. Tiền xử lý văn bản "
  "đáng giá hơn thêm một modality thứ hai.")
p("Các đối chứng giao thức không có gì đáng nói, đúng như thiết kế: XLM-R base cho 74,63, cửa sổ 128 token cho "
  "76,49, và một learning rate duy nhất cho 74,41 và 74,75. Lịch hai learning rate xứng đáng có chỗ, và backbone "
  "quan trọng hơn mục tiêu giám sát.")
p("Còn hai đối chứng phía đánh giá, và đây mới là những cái quan trọng. Split tách sản phẩm là một lần rút từ một "
  "cách chia ngẫu nhiên; hai cách chia nữa đưa mô hình phẳng về 75,22–75,94 và cascade về 73,98–77,37, nên điểm "
  "của cascade phụ thuộc vào cách chia tới gần ba điểm — lớn hơn giá trị của phân cấp một bậc độ lớn. Giữ nguyên "
  "toàn bộ category sản phẩm, sao cho không category nào trong test xuất hiện khi huấn luyện, cho 72,25 và 71,59 "
  "trên 5.800 dòng test. Càng xa cách chia gốc, con số càng thấp và thứ hạng càng kém ổn định.")
caption("Bảng IX. Các đối chứng, mỗi cái chạy lại so với tham chiếu của nó.")
table([
    ["Biến thể", "Seed", "Macro-F1", "Δ"],
    ["Đơn tác vụ (tham chiếu)", "3", "76,10 ± 0,71", "—"],
    ["  bỏ trùng văn bản", "3", "74,72 ± 1,34", "−1,38"],
    ["  bỏ gần trùng", "1", "73,66", "−2,44"],
    ["  chuẩn hoá teencode", "1", "77,56", "+1,46"],
    ["  XLM-R base", "3", "74,63 ± 1,55", "−1,47"],
    ["  128 token", "3", "76,49 ± 0,46", "+0,39"],
    ["  một learning rate", "3", "74,41 ± 0,37", "−1,69"],
    ["Đa tác vụ (tham chiếu)", "3", "76,02 ± 0,96", "—"],
    ["  bỏ trùng văn bản", "3", "77,02 ± 0,77", "+1,00"],
    ["  bỏ gần trùng", "1", "74,46", "−1,56"],
    ["  chuẩn hoá teencode", "1", "76,21", "+0,19"],
    ["  một learning rate", "3", "74,75 ± 1,65", "−1,27"],
    ["Fusion, encoder dùng chung", "3", "75,98 ± 1,15", "−0,12"],
    ["Fusion, encoder đóng băng riêng", "3", "75,95 ± 0,75", "−0,15"],
    ["Tách category, đơn tác vụ", "1", "72,25", "—"],
    ["Tách category, đa tác vụ", "1", "71,59", "—"],
])
h2("V.7. Hành vi theo lớp")
p("Bảng X, Hình 1 và Bảng XI phân rã tập test chính thức theo lớp và theo loại lỗi. Bức tranh không phải điều một "
  "bài bốn lớp gợi ra: nhầm loại rất hiếm. Ở mô hình đơn tác vụ, 226 review rác bị gán không rác và 106 review "
  "không rác bị gắn cờ rác, so với chỉ 36 trường hợp quyết định rác đúng nhưng loại sai; mô hình đa tác vụ đổi "
  "các con số này thành 208, 106 và 29. Mất mát tập trung ở hai loại rác nhỏ: 41% trong 54 dòng test Spam-1 và "
  "35% trong 310 dòng Spam-2 bị gán không rác, so với 11% trong 696 dòng Spam-3. Quyết định nhị phân do đó là nút "
  "thắt, và đầu loại hiếm khi có cơ hội sai.")
p("Các false positive đáng xem kỹ. Trong 106 review không rác mà mô hình đơn tác vụ gắn cờ rác, 9,4% chứa cụm "
  "“chất lượng sản phẩm tuyệt vời”, cụm xuất hiện trong 21,8% review Spam-1 của train nhưng chỉ 0,65% review "
  "không rác của train; trong 2.808 review không rác mô hình đoán đúng, 0,39% chứa cụm này. Không false positive "
  "nào là bản sao nguyên văn của một dòng huấn luyện. Mô hình đang phản ứng với một văn mẫu thực sự chỉ dấu "
  "Spam-1 trong bộ dữ liệu này, nên một phần tỉ lệ false positive phản ánh nhiễu nhãn chứ không phải lỗi mô hình "
  "— điều cần lưu ý khi diễn giải một macro-F1 ở mức giữa 70.")
p("Hai mô hình bất đồng trên 147 trong 3.974 dòng test: mô hình đa tác vụ sửa 86 và làm hỏng 61, thuần +25 dự "
  "đoán đúng. Các sửa đổi rải đều qua các lớp chứ không tập trung ở một lớp, và Bảng XI cho một ví dụ mỗi mẫu lặp "
  "lại, gồm cả trường hợp ngược khi cascade đọc ký tự kéo dài thành không phải review và gán sai một review bình "
  "thường.")
caption("Bảng XI. Kết quả theo lớp trên split chính thức (3 seed). Hỗ trợ là số dòng test của lớp.")
table([
    ["Lớp", "Hỗ trợ", "P (phẳng)", "R (phẳng)", "F1 (phẳng)", "P (cascade)", "R (cascade)", "F1 (cascade)"],
    ["Non-spam", "2.914", "0,928", "0,961", "0,944", "0,933", "0,955", "0,944"],
    ["Spam-1", "54", "0,605", "0,512", "0,553", "0,698", "0,457", "0,548"],
    ["Spam-2", "310", "0,701", "0,622", "0,659", "0,668", "0,645", "0,655"],
    ["Spam-3", "696", "0,932", "0,849", "0,889", "0,921", "0,868", "0,894"],
])
h2("V.8. Cái gì làm thay đổi metric")
p("Bảng X đặt mọi lựa chọn thiết kế lên một thang, lấy độ lệch seed làm thước. Các lựa chọn kiến trúc nằm ở "
  "không, backbone ở 1,5 điểm, và các lựa chọn đánh giá ở 1,4 đến 2,3. Nếu mục tiêu là một con số tốt hơn trên "
  "bài này, đổi encoder hoặc đổi cách đánh giá đáng giá hơn đổi cấu trúc giám sát một bậc độ lớn.")
caption("Bảng X. Lựa chọn nào làm dịch chuyển macro-F1. Δ đo so với tham chiếu tương ứng; "
        "độ lệch chuẩn seed 0,71 / 0,96 là thước đo.")
table([
    ["Nhóm", "Thay đổi", "Δ macro-F1"],
    ["Giám sát", "phẳng → cascade, split chính thức", "−0,08"],
    ["", "phẳng → cascade, bỏ trùng văn bản", "+2,30"],
    ["", "phẳng → cascade, tách sản phẩm", "−1,96"],
    ["Metadata", "nối category + mô tả", "−0,12"],
    ["Backbone", "PhoBERT → XLM-R", "−1,47"],
    ["Cửa sổ", "128 token", "+0,39"],
    ["Lịch", "hai learning rate → một", "−1,69 / −1,27"],
    ["Đánh giá", "split gốc → bỏ trùng, phẳng", "−1,38"],
    ["", "split gốc → bỏ trùng, cascade", "+1,00"],
    ["Nhiễu seed", "độ lệch chuẩn, phẳng / cascade", "±0,71 / ±0,96"],
], widths=[3.0, 8.5, 3.5])

h1("VI. Thảo luận")
p("Con số do cách đánh giá quyết định, không do kiến trúc. Bảng X làm so sánh tường minh. Trên split chính thức, "
  "hai mô hình cách nhau một phần mười điểm; cùng cặp đó cách nhau 2,3 điểm khi bỏ văn bản huấn luyện bị trùng, và "
  "2,0 điểm khi giữ sản phẩm ngoài train. Seed đầu tiên của chính chúng tôi kể câu chuyện thứ ba — 77,10 so với "
  "75,35 — đó là dạng một khẳng định một-split trông như thế nào trước khi được lấy trung bình. Mỗi cách đánh giá "
  "đều biện minh được và chúng mâu thuẫn nhau; một so sánh một-split, kể cả của chúng tôi, nên được đọc với điều "
  "đó trong đầu.")
p("Không đầu nào được hiệu chuẩn tốt. Expected calibration error top-label là 6,6% trên dev và 7,8% trên test cho "
  "mô hình phẳng, 6,0% và 5,3% cho cascade. Con số của cascade kém tin hơn, vì xác suất lớp của nó là tích hai "
  "marginal (công thức 2); không đầu ra nào nên dùng như xác suất mà không hiệu chuẩn lại.")
p("Vì sao phân cấp không trả công. Cascade áp một ràng buộc mà bộ phân loại phẳng vốn học được: lớp 1–3 không bao "
  "giờ đồng thời với không rác trong nhãn, và mô hình phẳng có 14.306 ví dụ để học đúng điều đó. Cái phân cấp mua "
  "được là một thiên lệch về quyết định cha, và phân rã lỗi cho thấy quyết định cha là nơi gần như toàn bộ mất mát "
  "đã nằm — 226 ca bỏ sót rác so với 36 ca nhầm loại. Chia sẻ một encoder giữa hai đầu không làm quyết định khó "
  "trở nên dễ hơn.")
p("Trùng lặp sản phẩm không chi phối kết quả, và cũng không cứu được phân cấp. Kiểm toán tìm thấy trùng lặp mức "
  "sản phẩm dày đặc trong split chính thức, nhưng bỏ nó chỉ mất ít với mô hình phẳng: 76,10 trên split chính thức "
  "so với 75,94 trên split tách. Cascade giảm nhiều hơn, từ 76,02 xuống 73,98. So sánh giữa hai split không sạch, "
  "vì tập test tách sản phẩm là một mẫu mới, nhưng thứ tự trong cùng split thì sạch, và trên sản phẩm mới mô hình "
  "phẳng dẫn trước. Số dòng trùng nguyên văn giải thích một phần vì sao split ngẫu nhiên hào phóng với cả hai; một "
  "cách đánh giá thực sự sạch sẽ thêm bước bỏ trùng mức văn bản lên trên việc chia theo nhóm, điều Bảng IX xấp xỉ.")
p("Kết quả âm là một phần của kết quả. Hai can thiệp hợp lý đã được chạy lại và không cái nào giúp. Đánh trọng số "
  "nghịch tần số làm metric tệ hẳn, đúng kiểu thất bại thường gặp khi một sơ đồ đánh trọng số được chỉnh cho một "
  "lớp chỉ có 54 dòng test; nối metadata, thứ chúng tôi kỳ vọng sẽ hại vì cắt cụt, lại nằm trong nhiễu so với "
  "baseline và do đó không nói gì về metadata như vậy. Báo cả hai tốn một trang và tiết kiệm cho người đọc sau một "
  "tuần.")

h1("VII. Đe dọa hiệu lực")
p("Hiệu lực khái niệm. Nhãn là chú thích của con người về một thuộc tính chủ quan: ranh giới giữa khen ngợi bịa "
  "đặt và sự hào hứng thật không phải lúc nào cũng quan sát được chỉ từ văn bản, và metric của chúng tôi thừa "
  "hưởng nhiễu đó.")
p("Hiệu lực nội tại. Mọi run chính dùng chung một cấu hình siêu tham số; các đối chứng ở Bảng IX thay đổi cửa sổ "
  "token và backbone, và không cái nào dịch chuyển điểm đáng kể, nhưng một lượt quét đầy đủ, gồm learning rate, "
  "vẫn ngoài phạm vi. Dừng sớm là nguồn phương sai thứ hai: split dev chỉ có 17 dòng Spam-1, nên epoch tối đa hoá "
  "macro-F1 trên dev khác nhau giữa các seed (3 đến 9 trong các run của chúng tôi) và một phần độ lệch seed là "
  "nhiễu dừng chứ không phải nhiễu mô hình. Padding động và bfloat16 thay đổi số học một chút so với các run thăm "
  "dò, nên những run đó được báo riêng và không trộn vào bảng.")
p("Hiệu lực ngoại tại. Nghiên cứu chỉ gồm một bộ dữ liệu, một ngôn ngữ và một backbone. Split tách sản phẩm là "
  "một đại diện cho triển khai, không phải mô phỏng triển khai: nền tảng thật gặp sản phẩm mới, văn mẫu mới và "
  "thích nghi đối kháng theo thời gian.")
p("Hiệu lực kết luận. Ba seed cho một ước lượng thô về phương sai, và lớp hiếm nhất có 54 dòng test, nên các con "
  "số theo lớp được báo như mô tả chứ không suy diễn.")

h1("VIII. Tái lập")
p("Repository chứa một script huấn luyện mọi mô hình được báo cáo ở đây, phần dựng cách chia cho đánh giá tách "
  "sản phẩm, các script kiểm toán, và file cấu hình, log, metric, dự đoán từng dòng của mọi run. Checkpoint mô "
  "hình không được gửi kèm, vì script huấn luyện tái tạo chúng từ cấu hình đã ghi. Seed, siêu tham số và phần "
  "cứng được liệt kê ở Bảng III. Phiên bản gói được pin trong requirements.txt, và mọi bảng, hình trong bài được "
  "sinh từ các file kết quả bởi một script phân tích duy nhất, nên một con số trong văn bản truy được về run đã "
  "tạo ra nó.")

h1("IX. Kết luận")
p("Chúng tôi xem lại phân loại review rác tiếng Việt với một giao thức khoá duy nhất, ba seed, một cách chia tách "
  "sản phẩm và một kiểm toán trùng lặp. Kết quả đã công bố tái lập được về bậc độ lớn nhưng không chính xác, vì "
  "bản công khai không phải snapshot đã công bố. Mục tiêu đa tác vụ phân cấp không cho cải thiện đo được so với "
  "bộ phân loại phẳng khi đã lấy trung bình qua seed, dù hai mô hình mắc lỗi khác nhau một cách hệ thống. Dưới "
  "cách chia tách sản phẩm, bộ phân loại phẳng dẫn trước khoảng hai điểm, nên phân cấp cũng không mua được độ "
  "bền. Phân rã lỗi chỉ về quyết định nhị phân và về nhiễu chú thích ở các review không rác mang văn mẫu, chứ "
  "không phải nhầm loại; và tách theo sản phẩm một mình không phải một phép thử sạch cho khái quát hoá.")

h1("Tài liệu tham khảo")
references = [
    "C. Van Dinh, S. T. Luu, and A. G.-T. Nguyen, “Detecting spam reviews on Vietnamese e-commerce websites,” in Proc. ACIIDS, 2022, pp. 595–607.",
    "C. Van Dinh and S. T. Luu, “Metadata integration for spam reviews detection on Vietnamese e-commerce websites,” Int. J. Asian Lang. Process., vol. 34, no. 1, 2024.",
    "N. Jindal and B. Liu, “Opinion spam and analysis,” in Proc. WSDM, 2008, pp. 219–230.",
    "M. Ott, Y. Choi, C. Cardie, and J. T. Hancock, “Finding deceptive opinion spam by any stretch of the imagination,” in Proc. ACL, 2011, pp. 309–319.",
    "A. Mukherjee, B. Liu, and L. Wang, “Spotting fake reviewer groups in consumer reviews,” in Proc. WWW, 2012, pp. 191–200.",
    "A. Mukherjee, V. Venkataraman, B. Liu, and N. Glance, “What Yelp fake review filter might be doing?” in Proc. ICWSM, 2013.",
    "S. Rayana and L. Akoglu, “Collective opinion spam detection: bridging ranking and classification,” in Proc. KDD, 2015, pp. 985–994.",
    "D. Q. Nguyen and A. T. Nguyen, “PhoBERT: Pre-trained language models for Vietnamese,” in Findings of EMNLP, 2020, pp. 1037–1042.",
    "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL-HLT, 2019, pp. 4171–4186.",
    "Y. Liu et al., “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv:1907.11692, 2019.",
    "A. Conneau et al., “Unsupervised cross-lingual representation learning at scale,” in Proc. ACL, 2020, pp. 8440–8451.",
    "X. Liu, P. He, W. Chen, and J. Gao, “Multi-task deep neural networks for natural language understanding,” in Proc. ACL, 2019, pp. 4487–4496.",
    "C. N. Silla and A. A. Freitas, “A survey on hierarchical classification across different application domains,” Data Mining and Knowledge Discovery, vol. 22, pp. 31–72, 2011.",
    "Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-balanced loss based on effective number of samples,” in Proc. CVPR, 2019.",
    "T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” in Proc. ICCV, 2017, pp. 2980–2988.",
    "A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, “Long-tail learning via logit adjustment,” in Proc. ICLR, 2021.",
    "J. M. Johnson and T. M. Khoshgoftaar, “Survey on deep learning with class imbalance,” J. Big Data, vol. 6, no. 1, p. 27, 2019.",
    "B. Efron, “Bootstrap methods: another look at the jackknife,” Ann. Statist., vol. 7, no. 1, pp. 1–26, 1979.",
    "T. G. Dietterich, “Approximate statistical tests for comparing supervised classification learning algorithms,” Neural Computation, vol. 10, no. 7, pp. 1895–1923, 1998.",
    "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” Psychometrika, vol. 12, no. 2, pp. 153–157, 1947.",
    "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017, pp. 1321–1330.",
]
for index, item in enumerate(references, 1):
    paragraph = doc.add_paragraph(f"[{index}] {item}")
    paragraph.paragraph_format.left_indent = Cm(0.8)
    paragraph.paragraph_format.first_line_indent = Cm(-0.8)
    paragraph.paragraph_format.space_after = Pt(2)
    for run in paragraph.runs:
        run.font.size = Pt(9)

doc.save(OUT)
print("saved", OUT)
