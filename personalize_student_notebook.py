from pathlib import Path

import nbformat


SOURCE = Path(r"C:\Users\ACER\Downloads\notebookaa871b3419.ipynb")
TARGET = Path(r"C:\Users\ACER\Downloads\Phan_Ba_Song_Toan_DE200247_SST2.ipynb")


def markdown(text):
    return nbformat.v4.new_markdown_cell(text)


nb = nbformat.read(SOURCE, as_version=4)

# Keep the original calculations and outputs, but add a short student header.
header = markdown(
    """# SST-2 Sentiment Analysis

**Student name:** Phan Bá Song Toàn  
**Student ID:** DE200247

Trong bài này mình thử hai phần chính: một baseline dùng TF-IDF kết hợp Logistic Regression, sau đó làm quen với tokenizer và DistilBERT. Mình giữ đúng cách chia dữ liệu theo đề để kết quả có thể kiểm tra lại được.
"""
)

notes = [
    markdown(
        """### 1. Load dữ liệu

Đầu tiên mình tải SST-2 và xem thử cấu trúc dataset cũng như một sample để biết tên các cột cần dùng.
"""
    ),
    markdown(
        """### 2. Chia dữ liệu

Mình lấy 1.200 câu từ train rồi chia thành 1.000 câu để train và 200 câu để development. 200 câu từ validation được giữ riêng để đánh giá cuối.
"""
    ),
    markdown(
        """### 3. Baseline

Mình chạy mô hình đơn giản trước để có mốc so sánh. TF-IDF biến câu thành các feature, còn Logistic Regression dự đoán nhãn positive hoặc negative.
"""
    ),
    markdown(
        """### 4. Tokenizer

Ở phần này mình kiểm tra một câu sau khi đi qua DistilBERT tokenizer. Các token và input id giúp mình thấy câu được chuyển thành dạng mà model có thể xử lý như thế nào.
"""
    ),
    markdown(
        """### 5. DistilBERT

Mình load checkpoint gốc với hai nhãn. Nếu classification head được khởi tạo mới thì đó là bình thường, vì phần head sẽ được học trong bước fine-tuning.
"""
    ),
    markdown(
        """### 6. Kiểm tra baseline trên evaluation set

Phần dưới đây là kết quả hiện tại của baseline. Các bước fine-tuning, confusion matrix và error analysis của DistilBERT cần bổ sung tiếp theo theo yêu cầu đề.
"""
    ),
]

# Insert a note immediately before the matching code cell, if possible.
new_cells = [header]
note_index = 0
for cell in nb.cells:
    source = "".join(cell.get("source", []))
    if cell.cell_type == "code" and "from sklearn.metrics import accuracy_score, f1_score" in source:
        cell.source = source.replace(
            "from sklearn.metrics import accuracy_score, f1_score",
            "from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix",
        )
    if cell.cell_type == "code" and note_index < len(notes):
        new_cells.append(notes[note_index])
        note_index += 1
    new_cells.append(cell)

nb.cells = new_cells
for cell in nb.cells:
    if not cell.get("id"):
        cell["id"] = nbformat.v4.new_code_cell().get("id")

nbformat.write(nb, TARGET)
print(TARGET)
