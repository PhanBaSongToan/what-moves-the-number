# Gói nộp conference — checklist

Tài liệu này liệt kê mọi việc phải làm trước khi nộp, theo thứ tự. Mỗi mục ghi rõ ai làm và mất bao lâu.

## 1. Chọn venue và đọc CFP (1 giờ)

- [ ] Xác định venue: workshop (Insights4NLP / Repl4NLP) hay hội nghị trong nước (VLSP, SoICT, KSE, RIVF).
- [ ] Tải template chính thức + đọc **page limit** (thường 8 trang *không* kể references với ACL; 6–8 trang với IEEE).
- [ ] Đọc chính sách **anonymity** (double-blind hay single-blind) và chính sách **preprint/arXiv**.
- [ ] Đọc **reproducibility checklist** của venue (nếu có).

## 2. Số liệu phải chốt trước khi viết (xem `experiments/`)

- [ ] Mọi ô trong bảng có **3 seed** — không còn ô 1 seed.
- [ ] Bootstrap CI cho mọi so sánh chính.
- [ ] Bộ dữ liệu thứ hai (Ott 2011) đã chạy xong, có bảng riêng.
- [ ] Kiểm tra độ nhạy ngưỡng near-duplicate (0.7 / 0.8 / 0.9).
- [ ] Calibration: temperature scaling đã fit trên dev, báo ECE trước/sau.

## 3. Viết lại bài (3–5 ngày)

- [ ] **Abstract**: nêu hai corpus, không chỉ một.
- [ ] **Introduction**: thêm một đoạn định vị so với evaluation studies (Dodge 2019, Lucic 2018, Bouthillier 2021, Reimers 2017).
- [ ] **Contributions**: liệt kê thành danh sách đánh số, mỗi đóng góp một dòng.
- [ ] **§V.7 Second corpus**: bảng + một đoạn so sánh hướng hiệu ứng giữa hai bộ.
- [ ] **Limitations**: bỏ "một dataset" (đã giải quyết), thay bằng hạn chế mới (ngôn ngữ, thời gian, human eval).
- [ ] **Related work**: thêm tiểu mục evaluation studies (đã có bản nháp trong `paper/paper.tex`).

## 4. Anonymize (nếu double-blind) — 1 giờ

- [ ] Xoá tên + affiliation trong `paper.tex` (`\author{...}` → `Anonymous submission`).
- [ ] Xoá link GitHub trong phần Reproducibility và các footnote.
- [ ] Xoá tên repo khỏi mọi hình/bảng.
- [ ] Tạo mirror ẩn danh: đẩy repo lên `anonymous.4open.science` và dùng link đó.
- [ ] Kiểm tra `git log` không lộ tên trong PDF (không sao, chỉ PDF quan trọng).
- [ ] Xoá phần acknowledgements (nếu có).

Script kiểm tra nhanh:

```powershell
Select-String -Path paper\paper.tex -Pattern 'github|Phan|Trinh|Tran Quang|Da Nang'
```

## 5. Đổi template (0.5–1 ngày)

- [ ] Chuyển `paper.tex` sang class của venue (nếu không phải IEEEtran).
- [ ] Chạy lại toàn bộ build, kiểm **page limit** và **overfull boxes**.
- [ ] Kiểm bảng không tràn cột; kiểm hình còn đọc được ở cỡ in.

## 6. Reproducibility checklist (2 giờ)

- [ ] Seed, hardware, hyperparameter: có trong bảng III.
- [ ] Code: link ẩn danh, có `README` mô tả 3 lệnh chạy.
- [ ] Dữ liệu: link công khai (Hugging Face + Ott 2011 gốc) và script dựng split.
- [ ] Thời gian chạy: ghi rõ tổng giờ GPU.
- [ ] Kết quả: file CSV từng dòng dự đoán cho mọi run.

## 7. Trước khi bấm nộp (1 giờ)

- [ ] Đọc toàn bài một lượt, sửa lỗi chính tả.
- [ ] Kiểm mọi `\ref` và `\cite` không bị `??` (xem `paper.log`).
- [ ] Chạy `python experiments/analyze_results.py` lần cuối, xác nhận `git diff` sạch.
- [ ] Build PDF lần cuối, kiểm số trang.
- [ ] Kiểm tra similarity bằng công cụ của trường (nếu có).
- [ ] Lưu bản PDF + source vào một thư mục riêng, đặt tên theo venue.

## 8. Sau khi nộp

- [ ] Lưu số submission.
- [ ] Chuẩn bị rebuttal: mỗi bảng một đoạn giải thích 3 câu.
- [ ] Nếu bị reject: chuyển sang workshop, dùng lại nguyên bản, chỉ đổi template.

## Việc có thể làm ngay hôm nay

| Việc | Thời gian | Phụ thuộc |
|---|---|---|
| Chạy xong 3 seed cho mọi ô | đang chạy | GPU |
| Chạy bộ dữ liệu thứ hai | 30 phút sau khi GPU rảnh | ở trên |
| Viết §V.7 | 1 giờ | số liệu |
| Anonymize | 1 giờ | — |
| Đọc CFP venue | 1 giờ | — |
