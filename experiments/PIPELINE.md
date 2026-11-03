# Pipeline của bài báo

Toàn bộ dự án chỉ có **một luồng** đi từ dữ liệu thô đến file PDF. Không có nhánh phụ.

```
data/vispamdetectionv2_official.zip          ← tải từ Hugging Face (một lần)
        │
        │  python experiments/build_product_disjoint_split.py [--seed N] [--by category]
        ▼
data/product_disjoint_split_seed42/          ← thêm seed123, seed456, category_42
        │
        │  python experiments/run_all.py
        ▼
experiments/results/<tên-run>/
        config + training_log + final_test_metrics + test_predictions + dev_predictions
        │
        │  (run_all.py tự gọi ở bước cuối, hoặc chạy riêng) python experiments/analyze_results.py
        ▼
paper/tables/*.tex   paper/figures/*.pdf   paper/numbers.tex
        │
        │  powershell -File paper/build.ps1
        ▼
paper/paper.pdf  →  paper_ieee_vispam.pdf
```

## Ba lệnh duy nhất

```powershell
# 1. chuẩn bị dữ liệu (một lần)
python experiments/build_product_disjoint_split.py
python experiments/build_product_disjoint_split.py --seed 123
python experiments/build_product_disjoint_split.py --seed 456
python experiments/build_product_disjoint_split.py --by category --seed 42

# 2. huấn luyện + sinh bảng biểu (tự bỏ qua run đã xong, có thể dừng giữa chừng)
python experiments/run_all.py

# 3. build PDF
powershell -ExecutionPolicy Bypass -File paper/build.ps1
```

## Bản đồ file — cái nào làm gì

| File | Vai trò | Bạn cần đọc khi nào |
|---|---|---|
| `experiments/config.py` | Tìm đường dẫn dữ liệu | khi đổi máy |
| `experiments/jobs.py` | **Danh sách 53 run** | khi muốn biết "chạy những gì" |
| `experiments/run_all.py` | Chạy danh sách đó | khi chạy |
| `experiments/run_experiment.py` | Định nghĩa mô hình, train, đánh giá | khi muốn biết "học thế nào" |
| `experiments/augment.py` | Chuẩn hoá teencode + near-duplicate | phần đối chứng |
| `experiments/analyze_results.py` | Sinh bảng, hình, `numbers.tex` | khi sửa bài |
| `experiments/experiment_stats.py` | Gom nhóm, phân rã lỗi | |
| `paper/paper.tex` | Bài báo; số liệu lấy từ `numbers.tex` | |
| `paper/build.ps1` | Build XeLaTeX 2 lần | |

Các script còn lại là **công cụ đo lường một lần** (audit trùng lặp, thống kê dataset, phân tích lỗi, lựa chọn ví dụ). Chúng không nằm trong pipeline huấn luyện; chạy khi cần con số cho một mục cụ thể.

## Một run là gì

Một "run" = một dòng trong `jobs.py`:

```python
dict(name="t2_multitask_dedup_s43", task=2, method="multitask",
     split="official", seed=43, dedupe=True)
```

Nó được truyền vào `run_experiment.Config`. Các trục thay đổi:

| Trục | Giá trị | Run tương ứng |
|---|---|---|
| `method` | `single` / `multitask` / `fusion` / `fusion2` | mô hình |
| `task` | 1 / 2 | nhị phân / 4 lớp |
| `split` | `official` / `disjoint` / `disjoint_123` / `disjoint_456` / `category_42` | cách chia |
| `seed` | 42 / 43 / 44 | seed |
| `dedupe` | `False` / `True` / `"near"` | bỏ trùng văn bản |
| `normalize` | `False` / `True` | chuẩn hoá teencode |
| `backbone` | PhoBERT / XLM-R | encoder |
| `max_len`, `encoder_lr`, `head_lr`, `type_loss`, `text_mode` | | các biến thể còn lại |

Mỗi run ghi ra một thư mục riêng và **không ghi đè** run khác; chạy lại thì bỏ qua.

## Vì sao trước đây trông rối

Pipeline cũ bị chia thành `run_matrix.py` (13 run), `run_extra.py` (8), `run_final.py` (13), `run_closure.py` (19) — vì các đợt thí nghiệm được thêm dần theo thời gian. Cùng một việc nhưng bốn chỗ khai báo. Giờ **chỉ còn `jobs.py`**; cả bốn file cũ đã xoá.

## Kiểm tra nhanh

```powershell
python experiments/run_all.py --list            # 53 run
python experiments/run_all.py --list --group main   # 6 run
python experiments/status.py                    # run nào đã xong
```

`analyze_results.py` **dừng với lỗi** nếu thiếu run, trừ khi đặt `ALLOW_MISSING=1` — để không bao giờ có bảng toàn số 0 trong bài.
