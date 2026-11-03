# Báo cáo kiểm tra dự án & bài báo TMG301

**Ngày kiểm tra:** 2026-11-03
**Phạm vi:** toàn bộ `C:\Users\ACER\Code\TMG301` (bản thảo `paper_ieee_vispam.tex`, PDF, toàn bộ script huấn luyện, artifact trong `%USERPROFILE%\Downloads\vispam_outputs`, các notebook bài tập khác)
**Phương pháp:** đọc mã nguồn, đối chiếu từng con số trong bài báo với file kết quả gốc, chạy lại kiểm tra split/leakage, thử build LaTeX.

> **Cập nhật:** báo cáo này mô tả bản nháp ngày 02/11. Toàn bộ lỗi P0/P1 đã được sửa trong cùng ngày 03/11 — xem **§11. Trạng thái sau khi sửa** ở cuối file, kèm số liệu mới và kết luận đã thay đổi.

---

## 0. Kết luận nhanh

**Bài báo CHƯA đạt chất lượng để nộp.** Không phải vì số liệu sai — phần lớn số liệu **khớp chính xác** với artifact — mà vì:

1. **So sánh trung tâm (baseline vs multi-task) bị nhiễu (confounded)**: hai mô hình chạy với protocol khác nhau, nên mức tăng +0.73 pp **không quy được cho kiến trúc multi-task**.
2. **Không thể tái lập**: bản thảo khai "cùng protocol", "lưu đủ config/log/metrics", nhưng script hiện tại không sinh ra được run chính; 2 run baseline không có file metric nào.
3. **Nội dung quá mỏng**: 2.350 từ, 4 tài liệu tham khảo, 1 hình, không có error analysis, không định nghĩa 3 loại spam.
4. **Bản PDF đang nộp là bản cũ và khác nguồn**: 1 tác giả (tex: 3 tác giả), template IEEEtran (tex: LNCS), và **tex hiện tại không compile được** (thiếu `llncs.cls`).

### Điểm theo tiêu chí

| Tiêu chí | Điểm | Ghi chú |
|---|---|---|
| Độ chính xác số liệu | 8/10 | Khớp artifact, trừ 2 run baseline không có file kết quả |
| Giá trị phương pháp | 4/10 | Confound protocol + 1 seed + thiếu baseline trên split disjoint |
| Khả năng tái lập | 3/10 | Script không sinh được kết quả chính; artifact nằm ngoài repo |
| Độ đầy đủ nội dung | 3/10 | 2.350 từ, 4 refs, thiếu hẳn 6–7 mục chuẩn |
| Trình bày / format | 4/10 | PDF lỗi thời, citation sai tác giả, bảng in đậm sai |
| Đóng góp học thuật ròng | 5/10 | Audit leakage là điểm sáng thật sự |

**Kết luận theo mức:** đạt mức "bài tập lớn trung bình–khá" nếu chỉ cần nộp môn; **không đạt** mức workshop/conference (cần 8–10 trang LNCS, 15+ refs, nhiều seed, error analysis).

---

## 1. Điểm mạnh (nên giữ)

- **Kiểm chứng được số liệu**: 14/16 con số trong bài khớp chính xác với file JSON artifact (xem §9). Đây là điều hiếm và đáng giữ.
- **Audit leakage sản phẩm là đóng góp thật**: 996/1.456/921 nhóm sản phẩm trùng giữa các split, và split disjoint thật sự disjoint (tôi đã kiểm lại: 0 nhóm trùng).
- **Trung thực với negative results**: báo cáo cả các biến thể làm giảm F1, có mục Limitations.
- **Không overclaim SOTA**: framing "auditable evaluation" là đúng hướng.

---

## 2. Lỗi nghiêm trọng (P0) — phải sửa trước khi nộp

### P0-1. Protocol của baseline và multi-task KHÔNG giống nhau → kết luận trung tâm mất hiệu lực

Bản thảo (`paper_ieee_vispam.tex:85`) khẳng định: *"encoder learning rate 1e-5, classification-head learning rate 5e-5 ... The same protocol is used for the single-task and multi-task primary comparisons."*

Thực tế:

| Tham số | Multi-task (`train_multitask.py`) | Baseline (`train_vispam_local.py`) |
|---|---|---|
| max_length | 100 | mặc định **128** |
| encoder LR | 1e-5 | **2e-5** (một LR duy nhất, hard-code `lr=2e-5`) |
| head LR | 5e-5 | không tồn tại khái niệm này |
| loss | CE | CE (dir `_ce`) nhưng script mặc định weighted |

→ Chênh lệch 0.73 pp có thể đến từ max_length/LR, **không phải từ multi-task**. Đây là lỗi mà reviewer chắc chắn bắt.

**Cách sửa:** chạy lại baseline với đúng max_len=100, encoder 1e-5/head 5e-5, CE. Chỉ cần 1 run (~25 phút GPU) là đủ để so sánh hợp lệ, hoặc 3 seed (~1.5 giờ).

### P0-2. Run chính không tái lập được bằng script hiện tại

`train_multitask.py:19` hard-code `OUTPUT_DIR = ...\multitask_type_weighted_task2_seed42` và `:129` hard-code `CrossEntropyLoss(weight=type_weights)`. Nghĩa là script hiện tại **chỉ sinh ra biến thể weighted** (kết quả 90.26/71.11), không sinh ra được run CE đã dùng cho kết quả chính `multitask_task2_seed42` (91.12/77.42).

Bằng chứng phụ: `config.json` của run CE **không có** key `"method"`, còn run weighted có → hai run do hai phiên bản script khác nhau tạo ra, phiên bản cũ đã mất.

### P0-3. Hai run baseline không có bất kỳ artifact nào

```
reproduction_official_v2_task1_ce_seed42\   → chỉ có model + tokenizer, KHÔNG có metrics.json
reproduction_official_v2_task2_ce_seed42\   → chỉ có model + tokenizer, KHÔNG có metrics.json
```

Nhưng bài viết (`:271`) khẳng định: *"All primary experiments ... retain their configuration, training log, best checkpoint, final metrics, and classification report."* → **sai sự thật**. Con số 91.55/89.04 và 91.37/76.69 (tức baseline của cả bài) không truy vết được.

### P0-4. PDF đang lưu không phải bản thảo hiện tại, và bản thảo hiện tại không build được

| | PDF (`paper_ieee_vispam.pdf`) | Nguồn (`paper_ieee_vispam.tex`) |
|---|---|---|
| Tác giả | **1 tác giả** (Phan Bá Song Toàn, DE200247) | **3 tác giả** (+ Trịnh Minh Bảo Trân, Trần Quang Long) |
| Template | IEEEtran (2 cột, "Index Terms", "TABLE I") | **LNCS** (`\documentclass[runningheads]{llncs}`) |
| Số trang | 4 | chưa build được |
| Mục | không có Reproducibility, Limitations gộp Future Work | đã tách |

Chạy `pdflatex` trên máy: `! LaTeX Error: File 'llncs.cls' not found`. `ieeetran.cls` thì có. → Bản PDF nộp kèm là bản cũ, thiếu 2 tác giả, khác template hoàn toàn.

**Cách sửa:** hoặc cài `llncs.cls` (MiKTeX) và build bản LNCS, hoặc quay lại IEEEtran cho khớp tên file. Phải chọn một và build lại.

### P0-5. Trích dẫn sai tác giả bài gốc

`paper_ieee_vispam.tex:41`: *"The paper by **Tran et al.**\cite{originalpaper}"* — nhưng `:278` là **Co Van Dinh and Son T. Luu**. Bản PDF cũ còn ghi `[1] T. M. B. Tran et al.` (sai cả trong bibliography).

Ngoài ra citation render dính chữ: `paper[1]`, `text[2]` (thiếu `~`), và `\cite` trong llncs nên dùng `~\cite{}`.

---

## 3. Lỗi lớn (P1)

### P1-1. Một seed duy nhất, không có CI
Chênh lệch 0.73 pp nằm trong biên độ dao động giữa các epoch (dev macro-F1 của chính run đó dao động 0.685→0.727). Bản thảo có thừa nhận (`:265`) nhưng **vẫn đưa vào abstract và kết luận như một phát hiện**. Cần 3–5 seed, báo mean ± std, tốt nhất là paired bootstrap test trên cùng test set.

### P1-2. So sánh product-disjoint bị confound bởi split
Bảng III so "random split 91.12" (split official) với "product-disjoint 90.96" (split tự tạo, train/dev/test khác hoàn toàn, kích thước khác). Ngoài ra **không có baseline single-task trên split disjoint** → không thể kết luận "multi-task bền vững với sản phẩm mới"; chỉ kết luận được "mô hình này chạy tốt trên split kia".

### P1-3. Bảng IV trộn hai protocol khác nhau
Các dòng "Category-only", "Description-only", "Separate fusion" chạy với **max_length=128 + weighted loss**, trong khi "Multi-task", "Rule-augmented", "Weighted type loss" chạy **max_length=100 + CE**. Đặt cạnh nhau trong một bảng ablation là không hợp lệ.

### P1-4. Bỏ sót kết quả quan trọng nhất về metadata
Original paper có kết quả metadata fusion; dự án **có** run metadata nhưng **không báo cáo**:

| Run | Acc | Macro-F1 | Ghi chú |
|---|---|---|---|
| `controlled_t2_metadata_weighted_seed42` | 89.91% | 74.40% | concat comment + category + description |
| `controlled_t1_metadata_weighted_seed42` | 90.99% | 88.82% | Task 1 |

Đây là phát hiện có giá trị: *concat text metadata làm giảm F1 so với comment-only*, trái với kết luận của bài gốc (dùng sentence-embedding fusion). Bỏ đi thì mất một đóng góp.

### P1-5. Nhãn 3 loại spam không được định nghĩa
Bài báo dùng class 1/2/3 xuyên suốt nhưng **không nói chúng là gì**. Theo bài gốc: SPAM-1 (fake review), SPAM-2 (review on brand only), SPAM-3 (non-review). Người đọc không thể hiểu per-class analysis.

### P1-6. Không có error analysis / confusion matrix / ví dụ định tính
Bài gốc có confusion matrix và error analysis; bài này chỉ có 1 câu "class 1 recall 0.49" rồi đẩy error analysis sang Future Work. Đây chính là chỗ làm bài "sơ sài" nhất.

### P1-7. Artifact không nằm trong repo + đường dẫn tuyệt đối hard-code
Toàn bộ kết quả nằm ở `C:\Users\ACER\Downloads\vispam_outputs\...`, mọi script hard-code path này; không có `requirements.txt`, không có README tái lập, không có git. Người khác (và chính bạn 1 tháng sau) không chạy lại được.

### P1-8. Chết code trong script
`run_controlled_experiment.py:180-183`: `classification_report(..., [max(...) for _ in range(0)]) if False else ...` — code vô nghĩa, để lại trong repo.

---

## 4. Lỗi nhỏ (P2)

| # | Vị trí | Vấn đề |
|---|---|---|
| 1 | `:187` Table II | In đậm **91.12%** dù **91.37%** cao hơn → đánh dấu "best" cho giá trị thấp hơn |
| 2 | `:228-229` | "Category-only control" thực chất là *comment + category*, không phải chỉ category → sai tên |
| 3 | `:27` | `\institute{TMG301 Text Mining, Vietnam}` — không phải tên trường/khoa |
| 4 | `:57` | "appropriate for the **course**" — ngôn ngữ bài tập, không phù hợp văn phong paper |
| 5 | Toàn bài | Trộn `10^{-5}` và `5\times10^{-5}`; "pp" viết tắt không định nghĩa |
| 6 | `:76` | "four-class supports" — nên nói rõ support là gì, và thêm bảng phân bố cả train/dev |
| 7 | `:144` | Liệt kê 12 biến thể "đã test" (ensemble, gated fusion, attentive pooling, logit adjustment, threshold tuning) nhưng **không có số nào** trong repo → không kiểm chứng được |
| 8 | `:33` | Abstract nói "limited robustness gap" khi so sánh bị confound (xem P1-2) |
| 9 | Tên file | `paper_ieee_vispam.*` nhưng template là LNCS |
| 10 | `:126-137` | Hình TikZ: nhãn "spam → type head" nhưng luồng `L_bin`/`L_type` vẽ chồng lên gate, dễ gây hiểu nhầm |

---

## 5. Thiếu gì để hết "sơ sài"

Hiện: **2.350 từ / 4 trang / 4 refs / 1 hình / 5 bảng**. Mức chuẩn LNCS: 8–12 trang, ~4.500–6.000 từ, 20–30 refs.

| Mục cần thêm | Ước lượng | Dữ liệu lấy từ đâu |
|---|---|---|
| Định nghĩa nhãn + ví dụ thật cho từng class | 0.5 trang | Dataset + bài gốc |
| Bảng thống kê dataset (số mẫu, vocab, độ dài TB, theo split) | 0.3 trang | Tính từ `vispamdetectionv2_official.zip` |
| Related work mở rộng (review spam: Jindal & Liu, Ott et al., Mukherjee et al.; multi-task; Vietnamese NLP) | 1.5 trang | Cần tra cứu + trích dẫn |
| Hypothesis rõ ràng + lý do multi-task giúp | 0.3 trang | Viết |
| Protocol table (đầy đủ hyperparameter của *từng* mô hình) | 0.3 trang | Config artifact |
| Multi-seed: mean ± std, và paired bootstrap | 0.5 trang | **Cần chạy GPU** |
| Confusion matrix (official + disjoint) | 0.5 trang | **Cần inference từ checkpoint có sẵn** |
| Error analysis: 15–20 ví dụ thật, phân loại lỗi | 1.5 trang | **Cần inference** |
| Báo cáo metadata fusion (đã có số) | 0.3 trang | Artifact có sẵn |
| Phân tích lỗi lan truyền của gate (binary sai → type sai) | 0.3 trang | **Cần inference** |
| Duplicate/near-duplicate audit (xem §6) | 0.3 trang | **Đã tính xong** |
| Threats to validity chi tiết (construct/internal/external) | 0.5 trang | Viết |
| Reproducibility: link repo, seed, phần cứng, thời gian chạy | 0.3 trang | Viết + đóng gói repo |

---

## 6. Phân tích mới tôi vừa chạy (có thể đưa ngay vào bài)

### 6.1 Split disjoint là hợp lệ
```
nhóm sản phẩm trùng: train/test = 0, train/dev = 0, dev/test = 0   ✓
kích thước: 13.985 / 1.947 / 3.938  (khớp bài: đúng)
nhãn test: 2961 / 49 / 284 / 644    (khớp bài: đúng)
```

### 6.2 Nhưng vẫn còn leak ở mức văn bản (bài chưa báo cáo)
```
36 dòng test có comment trùng nguyên văn với train (official split)
29 dòng test trùng nguyên văn train ngay trong split disjoint
201 dòng thuộc các nhóm comment trùng lặp trong toàn dataset
19.870 dòng, chỉ 19.757 comment khác nhau
```
→ Kể cả khi đã disjoint theo sản phẩm, **review spam được copy nguyên văn sang sản phẩm khác** vẫn lọt vào train. Đây là một phát hiện tốt, đúng chủ đề "robustness", và biến Future Work thành kết quả thực nghiệm.

### 6.3 Thống kê dataset công khai vs bài gốc
```
Bản công khai: 14.306 / 1.590 / 3.974, comment dài TB 26,19 từ (median 20, max 182), vocab 26.651
Bài gốc:       14.284 / 1.589 / 3.963, comment dài TB 17,38 từ
```
→ Chênh cả số mẫu **và** độ dài trung bình 8,8 từ. Đây là bằng chứng định lượng mạnh hơn cho lập luận "không phải cùng snapshot" — hiện bài chỉ nói suông.

---

## 7. Kế hoạch nâng cấp đề xuất

### Giai đoạn A — Sửa lỗi + bổ sung không cần GPU (~2 giờ)
1. Sửa P0-4 (build lại, chọn template), P0-5 (tác giả, `~\cite`).
2. Thêm: định nghĩa 3 loại spam + ví dụ; bảng thống kê dataset; bảng protocol theo từng mô hình.
3. Thêm: báo cáo metadata fusion; audit duplicate (§6.2); so sánh thống kê dataset (§6.3).
4. Viết lại Related Work (10–15 refs), Hypothesis, Threats to Validity.
5. Sửa P2 (in đậm, tên biến thể, institute, "course").

→ Kết quả: ~7 trang, không còn lỗi factual.

### Giai đoạn B — Sửa lỗi phương pháp (~2,5–3 giờ GPU)
1. Chạy baseline đúng protocol (max_len 100, 1e-5/5e-5, CE) — 1 run.
2. Chạy baseline + multi-task trên split disjoint — 2 run.
3. Nếu kịp: 3 seed cho 2 mô hình chính — +4 run.
4. Inference từ checkpoint có sẵn: confusion matrix, error analysis, gate error propagation (nhanh, ~10 phút).

→ Kết quả: so sánh hợp lệ, có mean ± std, có error analysis. Đây là phần quyết định chất lượng.

### Giai đoạn C — Đóng gói (~30 phút)
Repo con `paper/` gồm: tex, scripts, `requirements.txt`, README tái lập, `results/` chứa toàn bộ JSON/metrics, seed, log. Bỏ hard-code path tuyệt đối.

---

## 8. Vệ sinh dự án

Repo hiện là một thư mục hỗn hợp nhiều bài tập, không có git:

| Nhóm | File | Vấn đề |
|---|---|---|
| Bài báo | `paper_ieee_vispam.*`, 20 script `*.py` vispam | Không tách khỏi phần còn lại; artifact nằm ngoài |
| Bài tập NLP tích hợp | `Integrated_NLP_Amazon_Reviews*.ipynb` ×4 + 2 zip | 4 bản trùng lặp |
| SST-2 | `SST2_DistilBERT_*.ipynb` ×2, `build_sst2_transformer_notebook.py`, `final_distilbert/` (268 MB) | Trùng lặp |
| Bài tập khác | `ToanPBS*.ipynb`, `ngram_practice.ipynb`, `notebook5de6071c3d_fixed.ipynb`, `repair_translation_notebook.py` | Rời rạc |
| Slide | `build_opensearch_deck.mjs`, `output(s)/*.pptx`, `.opensearch-build/`, `tmp_baseline_build/` | 2 bộ build riêng |
| Rác | `tmp/`, `tmp_download.txt`, `tmp_original.pdf/txt`, `ipynb` (file rỗng tên lạ), 2 venv | ~1,4 GB |

Đề xuất: tạo git repo, thêm `.gitignore` (venv, model weights, tmp), tách `paper/`, `assignments/`, `slides/`.

---

## 9. Bảng truy vết số liệu (claim → artifact)

Đường dẫn artifact: `%USERPROFILE%\Downloads\vispam_outputs\`

| Số trong bài | Nguồn | Khớp |
|---|---|---|
| Official split 14.306/1.590/3.974 | `dataset_audit\product_leakage_report.json` | ✅ |
| Test supports 2.914/54/310/696 | cùng file | ✅ |
| Leakage 996/1.456/921 | cùng file | ✅ |
| Disjoint 13.985/1.947/3.938 | `product_disjoint_split_seed42\*.csv` (tôi đếm lại) | ✅ |
| Disjoint supports 2.961/49/284/644 | `multitask_product_disjoint_seed42\final_test_metrics.json` | ✅ |
| Published 89.23/85.58, 87.79/71.56 | `tmp_original.txt` L975–977, L1064 | ✅ |
| Baseline Task1 91.55/89.04 | *không có file* | ❌ |
| Baseline Task2 91.37/76.69 | *không có file* | ❌ |
| Multi-task Task2 91.12/77.42 | `multitask_task2_seed42\final_test_metrics.json` | ✅ |
| Multi-task Task1 91.95/89.51 | `multitask_task2_seed42\task1_binary_head_metrics.json` | ✅ |
| Disjoint 90.96/76.73 | `multitask_product_disjoint_seed42\final_test_metrics.json` | ✅ |
| Rule-augmented 91.02/75.12 | `rule_augmented_multitask_seed42\final_test_metrics.json` | ✅ |
| Weighted type loss 90.26/71.11 | `multitask_type_weighted_task2_seed42\final_test_metrics.json` | ✅ |
| Category-only 90.46/75.23 | `controlled_t2_category_weighted_seed42\metrics.json` | ✅ (nhưng sai tên) |
| Description-only 88.78/74.50 | `controlled_t2_description_weighted_seed42\metrics.json` | ✅ (sai tên) |
| Separate fusion 88.42/74.45 | `separate_fusion_t2_seed42\metrics.json` | ✅ |
| Per-class disjoint 0.94/0.56/0.67/0.89 | `multitask_product_disjoint_seed42\test_classification_report.txt` | ✅ |
| Metadata fusion (không báo cáo) | `controlled_t2_metadata_weighted_seed42\metrics.json` 89.91/74.40 | ⚠️ thiếu |

**Tỉ lệ kiểm chứng: 15/17 số khớp; 2 số cốt lõi (baseline) không có artifact.**

---

## 10. Việc cần bạn quyết

1. **Template:** giữ LNCS (cần cài `llncs.cls`) hay quay về IEEEtran (khớp tên file)?
2. **Mức đầu tư:** chỉ Giai đoạn A (sửa lỗi, ~2 giờ, vẫn 1 seed) hay A+B (chạy lại GPU ~3 giờ, sửa được lỗi phương pháp)?
3. **Ngưỡng nộp:** bài tập môn hay nhắm workshop/conference? (quyết định số seed và độ dài)

---

## 11. Trạng thái sau khi sửa — 2026-11-03 (đã hoàn tất)

Chọn hướng **IEEEtran** và **Giai đoạn A + B**. Bản mới nằm ở `paper/`, mã thực nghiệm ở `experiments/`, bản nháp cũ ở `archive/20261102_lncs_draft/`.

### 11.1 Đối chiếu lỗi

| Mã | Lỗi | Trạng thái |
|---|---|---|
| P0-1 | Baseline và multi-task chạy khác protocol | ✅ Một script duy nhất (`experiments/run_experiment.py`), một protocol khoá duy nhất (Bảng III của bài) |
| P0-2 | Script không sinh được run chính | ✅ `train_multitask.py` cũ đã chuyển vào `experiments/legacy/`; run mới tái lập được từ `run_matrix.py` |
| P0-3 | Baseline không có artifact | ✅ Mỗi run lưu `final_test_metrics.json`, `training_log.csv`, `test_predictions.csv`, `config` |
| P0-4 | PDF cũ, tex không build được | ✅ IEEEtran + XeLaTeX, 3 tác giả, 8 trang, build bằng `paper/build.ps1` |
| P0-5 | Trích dẫn sai tác giả (Tran et al.) | ✅ Đã sửa thành Dinh & Luu, dùng `~\cite` |
| P1-1 | 1 seed, không CI | ✅ 3 seed, mean ± sd, paired bootstrap 10k, McNemar exact |
| P1-2 | Disjoint thiếu baseline | ✅ Chạy cả single-task trên split disjoint |
| P1-3 | Bảng ablation trộn protocol | ✅ Hai ablation chạy lại dưới protocol khoá |
| P1-4 | Bỏ sót metadata fusion | ✅ Đã báo cáo, kèm giải thích cơ chế truncation (94,5% dòng vượt 100 token) |
| P1-5 | Không định nghĩa 3 loại spam | ✅ Bảng I: định nghĩa + ví dụ thật |
| P1-6 | Không có error analysis | ✅ Mục V-F, Bảng X (phân rã lỗi), Bảng XI (ví dụ), Hình 2 (confusion matrix) |
| P1-7 | Artifact ngoài repo, hard-code path | ✅ `experiments/results/`, README, requirements, biến môi trường cho đường dẫn dữ liệu |
| P1-8 | Dead code | ✅ Đã chuyển vào `legacy/` và ghi chú trong README của thư mục đó |

### 11.2 Kết quả mới — và kết luận đã đảo chiều

**34 run**, mỗi run 6–20 phút trên RTX 5050 Laptop:

| Cấu hình | Accuracy | Macro-F1 |
|---|---|---|
| Single-task PhoBERT, Task 2 (3 seed) | 90,89 ± 0,31 | **76,10 ± 0,71** |
| Multi-task PhoBERT, Task 2 (3 seed) | 90,87 ± 0,45 | **76,02 ± 0,96** |
| Binary chuyên dụng, Task 1 (3 seed) | 91,73 | 89,46 |
| Binary head của multi-task, Task 1 | 91,68 | 89,15 |
| Single-task, split disjoint (3 seed) | 90,99 | **75,94** |
| Multi-task, split disjoint (3 seed) | 90,67 | **73,98** |
| Single-task, bỏ 47 dòng train trùng văn bản (3 seed) | 90,14 ± 0,16 | 74,72 ± 1,34 |
| Multi-task, bỏ 47 dòng train trùng văn bản (3 seed) | 90,75 ± 0,31 | 77,02 ± 0,77 |
| Fusion: mã hoá mô tả ở lượt riêng (3 seed) | 90,86 ± 0,36 | 75,98 ± 1,15 |
| Single-task, backbone XLM-R base (3 seed) | 90,83 ± 0,13 | 74,63 ± 1,55 |
| Single-task, cửa sổ 128 token (3 seed) | 90,80 ± 0,42 | 76,49 ± 0,46 |
| Single-task, một learning rate 2e-5 | 90,87 | 74,56 |
| Multi-task, một learning rate 2e-5 | 90,84 | 73,19 |
| Multi-task + weighted type loss | 89,43 | 72,91 |
| Single-task + metadata concat | 90,66 | 76,66 |
| TF-IDF + Linear SVM (tham chiếu) | 84,40 | 66,52 |

**Kết luận quan trọng: mức tăng +0,73 pp của bản cũ là nhiễu seed.** Với 3 seed, chênh lệch là −0,08 pp (bootstrap CI [−0,61; +3,94], p = 0,142). Seed 42 cho +1,75 pp, seed 43 và 44 đảo lại. Bài mới viết lại toàn bộ abstract/kết luận theo hướng trung thực: đóng góp nằm ở protocol, audit leakage/duplicate và phân rã lỗi, **không** phải ở kiến trúc multi-task.

Các thí nghiệm kiểm soát cho thấy thứ hạng **phụ thuộc vào cách dựng evaluation**: trên sản phẩm mới single-task hơn 1,96 pp; sau khi bỏ các dòng train trùng văn bản multi-task lại hơn 2,30 pp. Ngược lại, XLM-R và learning-rate đơn làm giảm điểm của cả hai mô hình, còn fusion mô tả (không còn truncation) không thêm gì. Đây là thông điệp chính của bài: không thể kết luận về kiến trúc nếu chưa cố định evaluation — và mọi so sánh một split, kể cả của bài này, đều phải đọc theo tinh thần đó.

### 11.3 Phát hiện mới được đưa vào bài

- **Trùng lặp nguyên văn**: 38 dòng test trùng comment với train (35 văn bản khác nhau); 30 dòng vẫn trùng ngay trong split disjoint; 213/19.870 dòng thuộc nhóm trùng.
- **Vì sao metadata concat thất bại**: mô tả sản phẩm dài trung vị 315 token, 94,5% dòng vượt cửa sổ 100 token → thực chất là thí nghiệm về truncation, không phải về metadata.
- **Lỗi chủ yếu là bỏ sót spam, không phải nhầm loại**: 226 FN so với 36 nhầm loại (single-task, seed 42).
- **Label noise định lượng được**: 9,4% false positive chứa cụm "chất lượng sản phẩm tuyệt vời" (so với 0,39% ở các dòng non-spam đoán đúng).

### 11.4 Dựng lại từ đầu

```powershell
cd experiments
pip install -r requirements.txt
python run_matrix.py        # 13 run, ~2 giờ trên GPU 8 GB
python analyze_results.py   # sinh bảng, hình, numbers.tex
cd ..\paper
powershell -ExecutionPolicy Bypass -File build.ps1
```

Lưu ý: bản dịch PDF phải dùng **XeLaTeX** (không phải pdfLaTeX) vì bài có trích dẫn tiếng Việt; `llncs.cls` không cần nữa. Checkpoint (~540 MB/run) đã xoá để gọn repo, script sẽ sinh lại.

