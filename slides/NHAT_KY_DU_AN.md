# Nhật ký dự án — chuyện đã xảy ra theo thứ tự

Tài liệu này kể lại **quá trình làm**, không phải nội dung bài. Dùng khi thầy hỏi "em làm gì trong đồ án này" hoặc "quá trình nghiên cứu diễn ra thế nào".

---

## Giai đoạn 0 — Xuất phát điểm: một bản nháp có lỗi

Nhóm bắt đầu từ bài gốc của Dinh & Luu (2024): PhoBERT đạt **71,56 macro-F1** (Task 2) và **85,58** (Task 1) trên một split ngẫu nhiên của ViSpamReviews v2.

Bản nháp đầu của nhóm có 4 trang, báo cáo:
- baseline PhoBERT: 91,37 accuracy / 76,69 macro-F1
- mô hình đa tác vụ: 91,12 / **77,42**
- kết luận: "đa tác vụ cải thiện **+0,73** điểm"

Sau đó nhóm tự kiểm tra lại và phát hiện ba vấn đề:
1. Baseline và multi-task chạy **hai protocol khác nhau** (max_len 128 vs 100; một LR 2e-5 vs hai LR 1e-5/5e-5) → con số +0,73 không quy được cho kiến trúc.
2. Hai con số chính **không có file kết quả** nào lưu lại.
3. File PDF nộp kèm là bản cũ 1 tác giả, còn source là bản khác 3 tác giả và không build được.

**Quyết định:** dựng lại toàn bộ từ đầu, không sửa vá.

---

## Giai đoạn 1 — Dựng lại thước đo cho công bằng

Việc làm:
- Viết **một** script huấn luyện duy nhất, mọi mô hình đi qua nó.
- Khoá một protocol: PhoBERT-base, max_len 100, batch 32, LR encoder 1e-5 / head 5e-5, 10 epoch dừng sớm, seed 42–43–44.
- Chạy **13 run đầu**: 6 run so sánh chính, 3 run Task 1, 2 run split sản phẩm, 2 ablation.

Kết quả: **76,10 ± 0,71** (phẳng) và **76,02 ± 0,96** (đa tác vụ).
→ Chênh lệch còn **−0,08**, khoảng tin cậy chứa 0.

**Nghĩa là:** kết luận "+0,73" của bản nháp biến mất khi đo lại cho đúng.

---

## Giai đoạn 2 — Kiểm tra xem kết quả âm có đáng tin không

Nhóm tự đặt câu hỏi ngược: "hay là do nhóm cấu hình sai nên mô hình đa tác vụ mới không hơn?"

Việc làm: chạy thêm các đối chứng.
- Bỏ 47 dòng train trùng văn bản với test → huấn luyện lại.
- Đổi backbone sang XLM-R.
- Đổi cửa sổ 100 → 128 token.

Kết quả:
- Bỏ trùng: phẳng **74,72**, đa tác vụ **77,02** → đảo chiều.
- XLM-R: **74,63** (kém PhoBERT 1,47).
- 128 token: 76,49 (trong nhiễu).

**Nghĩa là:** kết quả không phải lỗi cấu hình; nhưng đổi cách đánh giá thì kết luận đổi.

---

## Giai đoạn 3 — Nhận ra vấn đề thật sự là cách đo

Ba con số đặt cạnh nhau:

| Cách đánh giá | Phẳng | Đa tác vụ | Kết luận |
|---|---|---|---|
| Split gốc (3 seed) | 76,10 | 76,02 | hòa |
| Tách theo sản phẩm | **75,94** | 73,98 | phẳng hơn 1,96 |
| Bỏ trùng văn bản | 74,72 | **77,02** | đa tác vụ hơn 2,30 |

Cùng hai mô hình, ba kết luận khác nhau.

**Quyết định:** đổi khung cả bài. Tiêu đề thành *"What Moves the Number?"* — câu hỏi không còn là "mô hình nào tốt hơn" mà là "cái gì làm con số thay đổi".

---

## Giai đoạn 4 — Giải thích vì sao

Ba việc, để không dừng ở "số nó thế":
1. **Phân rã lỗi:** 226 ca bỏ sót rác, 106 ca báo nhầm, chỉ **36** ca nhầm loại rác → nút thắt là quyết định nhị phân, không phải tầng phân loại.
2. **Nhiễu nhãn:** 9,4% false positive chứa cụm "chất lượng sản phẩm tuyệt vời" (21,8% review Spam-1; 0,65% non-spam) → mô hình học đúng tín hiệu nhưng bị nhãn phạt.
3. **Metadata:** nối text thất bại vì 94,5% dòng vượt 100 token (mô tả dài trung vị 315 token); làm bản mã hoá riêng, không cắt cụt → vẫn **75,98** → metadata thật sự không giúp.

---

## Giai đoạn 5 — Đóng gói

- Bài 8 trang IEEEtran, 24 tài liệu, build bằng XeLaTeX.
- 34 run, mỗi run lưu config, log, metrics, dự đoán từng dòng.
- Repo công khai + CI kiểm bảng khớp kết quả + tự build PDF.
- Slide 22 trang cho buổi báo cáo.

---

## Giai đoạn 6 — Đóng nốt các hạn chế (đang chạy)

19 run bổ sung, đã xong 41/53 tổng:
- **Phương sai giữa các cách chia:** cùng seed 42, partition 123 cho đa tác vụ **74,50**, partition 456 cho **77,37** → **nhảy 2,87 điểm chỉ vì đổi partition**, lớn hơn mọi hiệu ứng kiến trúc.
- **Fusion encoder riêng (fusion2):** 75,74 / 76,78 / 75,32 → TB **75,95**, vẫn không hơn.
- Còn chạy: near-duplicate, chuẩn hoá teencode, category-disjoint, seed cho learning-rate, calibration trên dev.

---

## Bản 60 giây (nếu thầy hỏi "quá trình làm thế nào")

> "Nhóm bắt đầu bằng việc tái lập bài gốc, nhưng khi kiểm tra lại thì phát hiện bản nháp của mình chạy hai protocol khác nhau, nên con số cải tiến không đáng tin. Nhóm dựng lại toàn bộ với một protocol duy nhất và 3 seed; kết quả là mô hình đa tác vụ không hơn gì. Để chắc không phải do cấu hình, nhóm chạy thêm các đối chứng về dữ liệu và backbone. Chính các đối chứng đó cho thấy đổi cách đánh giá thì kết quả đảo chiều, nên nhóm đổi hướng bài thành câu hỏi 'cái gì làm thay đổi điểm'. Cuối cùng nhóm giải thích cơ chế bằng phân rã lỗi và nhiễu nhãn, rồi đóng gói toàn bộ để tái lập được."

---

## Bảng nếu bị hỏi "vì sao lại làm bước đó"

| Bước | Câu hỏi tự đặt | Nếu không làm thì sao |
|---|---|---|
| Khoá protocol | "So sánh có công bằng?" | Không biết +0,73 đến từ đâu |
| 3 seed | "Một seed may mắn?" | Seed 42 cho +1,75 và ta kết luận sai |
| Tách sản phẩm | "Dữ liệu có rò rỉ?" | Điểm bị thổi phồng mà không biết |
| Bỏ trùng văn bản | "Rò rỉ còn dạng khác?" | Không phát hiện 30 dòng test trùng train |
| Đổi backbone | "Kết quả có phụ thuộc PhoBERT?" | Tưởng kiến trúc quan trọng hơn encoder |
| Phân rã lỗi | "Vì sao chỉ 76 điểm?" | Không biết nên cải thiện chỗ nào |
| Nhiễu nhãn | "Nhãn có sạch?" | Đổ lỗi cho mô hình thay vì dữ liệu |
