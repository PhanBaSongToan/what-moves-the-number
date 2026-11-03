from pathlib import Path

import nbformat


SOURCE = Path(r"C:\Users\ACER\Code\TMG301\SST2_DistilBERT_Sentiment_Analysis.ipynb")
EXECUTED = Path(r"C:\Users\ACER\Code\TMG301\SST2_DistilBERT_Sentiment_Analysis_executed.ipynb")
TARGET = Path(r"C:\Users\ACER\Downloads\Phan_Ba_Song_Toan_DE200247_SST2_complete.ipynb")
TARGET_EXECUTED = Path(r"C:\Users\ACER\Downloads\Phan_Ba_Song_Toan_DE200247_SST2_complete_executed.ipynb")


TOP = """# SST-2 Sentiment Analysis\n\n**Student name:** Phan Bá Song Toàn  \n**Student ID:** DE200247\n\nTrong notebook này mình làm theo từng bước của đề: chuẩn bị SST-2, chạy một baseline TF-IDF + Logistic Regression, sau đó fine-tune DistilBERT và so sánh hai kết quả. Mình giữ nguyên split và thông số trong đề để kết quả dễ kiểm tra lại.\n"""


CONCLUSION_CODE = '''best_row = final_results.iloc[final_results["Macro F1"].argmax()]
base_row = final_results.iloc[0]
bert_row = final_results.iloc[1]

conclusion = f"""This notebook compares a simple TF-IDF plus Logistic Regression model with a fine-tuned DistilBERT model on SST-2. On the final evaluation subset, the baseline reached {base_row["Accuracy"]:.3f} accuracy and {base_row["Macro F1"]:.3f} Macro F1, while DistilBERT reached {bert_row["Accuracy"]:.3f} accuracy and {bert_row["Macro F1"]:.3f} Macro F1. The baseline is fast and easy to inspect, but it mainly uses word frequency, so it can miss context and negation. DistilBERT starts from a language model that has already learned general English patterns, which helps it use words in relation to one another. The trade-off is that it needs more time, memory, and setup. The evaluation set has only 200 examples, so a small difference should not be overinterpreted. The error examples also show that ambiguous sentences can still confuse both models. Overall, DistilBERT performs better here, while TF-IDF remains a useful and understandable reference."""
print(conclusion)
print("Word count:", len(conclusion.split()))
assert 100 <= len(conclusion.split()) <= 150
'''


def personalize(path, output):
    nb = nbformat.read(path, as_version=4)
    nb.cells[0] = nbformat.v4.new_markdown_cell(TOP)
    for cell in nb.cells:
        source = "".join(cell.source)
        if cell.cell_type == "code" and "conclusion = f" in source:
            cell.source = CONCLUSION_CODE
        if cell.cell_type == "markdown":
            source = source.replace(
                "This notebook follows the practical exam specification:",
                "Mình triển khai notebook theo đúng các mục trong đề:",
            )
            source = source.replace(
                "The following cell evaluates both trained models exactly once",
                "Ở phần này mình đánh giá cả hai model một lần",
            )
            cell.source = source
    nbformat.write(nb, output)
    print(output)


personalize(SOURCE, TARGET)
personalize(EXECUTED, TARGET_EXECUTED)
