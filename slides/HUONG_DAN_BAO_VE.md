# Hiểu toàn bộ dự án — tài liệu cho leader

Mục đích: bạn đọc xong tài liệu này thì giải thích được **mọi lựa chọn** và **mọi con số** trong bài, kể cả khi bị hỏi ngược.

---

## 0. Một đoạn tóm tắt để mở đầu bất kỳ cuộc giải thích nào

> Nhóm tái lập bài toán phân loại review rác tiếng Việt trên bộ ViSpamReviews v2, nhưng thay vì chỉ báo một con số, nhóm chạy **34 cấu hình dưới cùng một protocol** để đo **yếu tố nào thực sự làm thay đổi điểm macro-F1**. Kết quả: thay đổi kiến trúc (đa tác vụ, thêm metadata) chỉ làm đổi **~0,1 điểm**; đổi backbone **1,5 điểm**; đổi **cách xây dựng tập đánh giá** thì **1,0–2,3 điểm** và đủ để **đảo thứ hạng** hai mô hình. Nhiễu seed đã là 0,7–1,0. Nên câu hỏi "mô hình nào tốt hơn" không trả lời được nếu chưa cố định cách đo.

---

## 1. Dữ liệu — bạn phải thuộc 6 con số

| Thứ | Số | Ý nghĩa |
|---|---|---|
| Kích thước | **14.306 / 1.590 / 3.974** | train / dev / test |
| Nhãn test | **2.914 / 54 / 310 / 696** | non-spam / Spam-1 / Spam-2 / Spam-3 |
| Lớp hiếm nhất | **211 dòng train (1,5%)** | Spam-1 |
| Độ dài | **trung vị 20 token** (mean 26,2; p95 68) | vì sao cửa sổ 100 là đủ |
| Từ vựng | 26.651 (train) | |
| Trùng lặp toàn bộ | **19.870 dòng / 19.750 văn bản** | ~120 dòng là bản sao |

**Bốn nhãn** (nhớ ví dụ, không nhớ định nghĩa):
- Non-spam: review có thông tin sản phẩm.
- Spam-1: review giả — khen/dìm có chủ đích. **Đây là lớp khó nhất** vì câu trôi chảy, tích cực, đúng ngữ pháp.
- Spam-2: chỉ nói về shop/thương hiệu/giao hàng, không nói chất lượng sản phẩm.
- Spam-3: không phải review — gõ loạn, lặp ký tự, emoji.

**Vì sao dùng macro-F1:** đoán tất cả là non-spam đã được 73,5% accuracy. Macro-F1 cho mỗi lớp trọng số bằng nhau, nên nó phản ánh việc học lớp hiếm.

---

## 2. Hai lỗ hổng của split ngẫu nhiên — trái tim của phần "dữ liệu"

### Lỗ hổng 1: rò rỉ theo sản phẩm
Split ngẫu nhiên theo *review*, nên **cùng một sản phẩm** nằm ở cả train và test. Mô hình học thuộc từ vựng sản phẩm (tên model, tên shop, mô tả) thay vì học dấu hiệu spam.

Cách đo: gộp `product_name + category + description`, chuẩn hoá (NFKC, lowercase, gộp khoảng trắng) thành một khoá sản phẩm, rồi đếm khoá xuất hiện ở nhiều split.

Số: **1.676 / 1.006 / 1.479** khoá duy nhất; trùng **996** (train–dev), **1.456** (train–test), **921** (dev–test).

### Lỗ hổng 2: copy-paste nguyên văn
Tách theo sản phẩm vẫn **không** xoá hết: review spam được copy sang sản phẩm khác. So khớp sau khi gộp khoảng trắng + lowercase:
- **38 dòng test** của split gốc trùng văn bản với train (35 văn bản khác nhau);
- sau khi tách theo sản phẩm vẫn còn **30 dòng**;
- **213 dòng** nằm trong các nhóm trùng lặp.

**Cách xử lý:** nhóm không xoá lặng lẽ — nhóm **giữ nguyên** và báo cáo số, rồi làm thêm **thí nghiệm đối chứng**: bỏ 47 dòng train trùng văn bản với test rồi huấn luyện lại.

### Split sản phẩm được dựng thế nào
Pool toàn bộ 19.870 dòng → `GroupShuffleSplit` lần 1 giữ 20% **nhóm sản phẩm** làm test → lần 2 chia phần còn lại thành train/dev (12,5%). Kết quả **13.985 / 1.947 / 3.938**, test **2.961/49/284/644**, **0 nhóm trùng**.

---

## 3. Protocol — vì sao phải khoá

Trước khi sửa, baseline và mô hình đa tác vụ chạy **hai protocol khác nhau** (max_len 128 vs 100; một LR 2e-5 vs hai LR 1e-5/5e-5). Như vậy chênh lệch 0,73 điểm có thể do protocol, không do kiến trúc. Đó là lỗi nghiêm trọng nhất của bản nháp.

Protocol khoá (mọi run trong bài):

| Thành phần | Giá trị |
|---|---|
| Encoder | `vinai/phobert-base-v2` (135M) |
| max_len | 100 token |
| Batch | 32, padding động theo batch (bội số 8) |
| Optimizer | AdamW, weight decay 0,01 |
| LR | encoder 1e-5 · head 5e-5 |
| Gradient clip | 1,0 |
| Precision | bfloat16 |
| Epoch | tối đa 10, dừng sớm theo dev macro-F1, patience 3 |
| Seed | 42, 43, 44 |

**Vì sao hai LR:** encoder đã được pretrain nên cần LR nhỏ; head khởi tạo ngẫu nhiên nên cần LR lớn hơn.

**Vì sao padding động:** chỉ thay đổi compute, không đổi bài toán (vị trí pad bị mask trong attention).

**Một hạn chế trung thực:** dừng sớm theo dev macro-F1, mà dev chỉ có 17 dòng Spam-1, nên epoch tốt nhất nhảy 3–9 giữa các seed. Một phần "nhiễu seed" thực ra là nhiễu dừng. Đã ghi trong Threats.

---

## 4. Ba mô hình — khác nhau đúng một chỗ

**Phẳng (baseline).** PhoBERT + một lớp linear 4 đầu ra. Cross-entropy 4 lớp.

**Đa tác vụ (cascade).** Cùng encoder, **hai** đầu ra:
- head nhị phân (spam / không spam);
- head 3 lớp (loại spam), **chỉ học trên review spam**.

Loss: `L = L_bin + 1[y_bin=1] · L_type`.
Dự đoán: nếu binary nói "không spam" → lớp 0; ngược lại lấy argmax của head loại + 1.

**Fusion (metadata).** Mã hoá thêm mô tả sản phẩm rồi ghép với vector review. Có 2 bản:
- `fusion`: encoder **dùng chung**, mô tả đi qua cùng encoder ở lượt forward thứ hai → 76,0 (không hơn gì);
- `fusion2` (đang chạy): encoder **đóng băng riêng** cho mô tả → 75,7–76,8.

**Điểm phải nói rõ khi bị hỏi:** đây **không phải** S-PhoBERT hai tầng của bài gốc. Bài gốc fine-tune một sentence encoder ở tầng 1 rồi mới dùng ở tầng 2; nhóm không tái lập tầng 1 (không có checkpoint đó), nên chỉ so sánh được ở mức "metadata có giúp không", không phải "tái lập bài gốc".

---

## 5. Ma trận thực nghiệm — biết mỗi run để làm gì

34 run trong bài, chia 6 nhóm:

| Nhóm | Run | Trả lời câu hỏi |
|---|---|---|
| Chính | single ×3, multi ×3 | đa tác vụ có hơn? |
| Task 1 | single binary ×3 (+ head nhị phân lấy từ run multi) | đa tác vụ có làm hỏng nhánh nhị phân? |
| Disjoint | single ×3, multi ×3 | sản phẩm mới thì sao? |
| Đối chứng | dedupe 47 dòng ×2×3 seed, XLM-R ×3, 128 token ×3, LR đơn ×2 | kết luận có phụ thuộc lựa chọn phụ? |
| Ablation | weighted loss, metadata concat (seed 42) | can thiệp hợp lý có giúp? |
| Tham chiếu | TF-IDF + LinearSVM | mô hình từ vựng được bao nhiêu? |

Ngoài ra còn **19 run đang chạy** để đóng nốt các hạn chế (near-duplicate, chuẩn hoá teencode, category-disjoint, 2 partition mới, seed cho LR, calibration trên dev). Chúng **chỉ thêm**, không làm sai số nào ở trên.

---

## 6. Kết quả — 8 con số và cách đọc

| So sánh | Số | Cách đọc |
|---|---|---|
| Phẳng vs đa tác vụ (split gốc, 3 seed) | **76,10 ± 0,71** vs **76,02 ± 0,96** | chênh −0,08; **CI [−0,61; +3,94] chứa 0** → không kết luận được |
| McNemar | **p = 0,047** | hai mô hình sai **khác chỗ**, không phải mô hình nào tốt hơn |
| Sửa / làm hỏng | **86 / 61** | net +25 câu trên 3.974 |
| Disjoint (3 seed) | **75,94** vs **73,98** | trên sản phẩm mới, phẳng **hơn 1,96** |
| Bỏ trùng văn bản (3 seed) | **74,72** vs **77,02** | đa tác vụ **hơn 2,30** |
| Backbone XLM-R | **74,63** | kém PhoBERT 1,47 → backbone quan trọng hơn objective |
| 128 token | **76,49** | trong nhiễu → cửa sổ không phải hyperparameter sống |
| Trọng số cân bằng lớp | **72,91** | tệ hơn 4,19 so với chuẩn seed 42 |

**Nhiễu seed** là ±0,71 (phẳng) và ±0,96 (đa tác vụ) — **lớn hơn hiệu ứng kiến trúc**. Đây là lý do phải chạy 3 seed; nếu chỉ một seed, nhóm đã từng báo +1,75 điểm và kết luận sai.

**Phân rã lỗi (seed 42):** 226 rác→không rác, 106 không rác→rác, chỉ **36 nhầm loại**. Đa tác vụ: 208 / 106 / 29. Kết luận: nút thắt là **quyết định nhị phân**, không phải tầng loại.

**Nhiễu nhãn:** 9,4% false positive chứa cụm "chất lượng sản phẩm tuyệt vời" (21,8% review Spam-1 train; 0,65% non-spam train; 0,39% ở ca đoán đúng).

---

## 7. Giải thích "vì sao" cho từng phát hiện

**Vì sao đa tác vụ không giúp?** Cascade áp một ràng buộc mà mô hình phẳng **tự học được**: lớp 1–3 không bao giờ đi với non-spam, và nó có 14.306 ví dụ để học điều đó. Cái cascade thêm là **thiên lệch về quyết định cha**, mà quyết định cha đã là chỗ mất điểm nhiều nhất (226 lỗi). Chia sẻ encoder không làm quyết định khó trở nên dễ hơn.

**Vì sao trên sản phẩm mới cascade tệ hơn?** Nhóm chỉ báo cáo **mô tả**, không khẳng định cơ chế: 2/3 seed phẳng thắng, khoảng cách ~2 điểm. Giả thuyết hợp lý: head nhị phân của cascade dựa nhiều hơn vào ngữ cảnh sản phẩm, nên khi ngữ cảnh mất thì rơi mạnh hơn. Nếu bị hỏi sâu: "đây là quan sát mô tả, chưa đủ để kết luận cơ chế".

**Vì sao bỏ trùng văn bản lại giúp cascade?** Mô hình phẳng hưởng lợi từ việc **nhớ văn mẫu**; xoá mẫu thì nó mất điểm nhiều hơn (−1,38 so với +1,00). Đây là lập luận có định hướng, 1 seed/nhánh — phải nói rõ.

**Vì sao metadata không giúp?** Bản nối text thất bại vì mô tả dài trung vị **315 token**, 94,5% dòng vượt cửa sổ 100 → thực chất là thí nghiệm về cắt cụt. Bản fusion mã hoá riêng (không cắt cụt) cũng chỉ 75,98 → **metadata thật sự không thêm tín hiệu** cho bài toán này.

**Vì sao trọng số cân bằng lớp làm tệ?** Nó phạt mạnh lớp 54 dòng, đẩy mô hình đoán bừa sang lớp hiếm và mất điểm ở 2 lớp lớn. F1 macro tụt 4,19.

**Vì sao XLM-R thấp hơn?** PhoBERT được pretrain trên tiếng Việt với tách từ phù hợp; XLM-R dùng sentencepiece đa ngữ, ít đặc thù tiếng Việt hơn. Hệ quả: đừng quy kết kết quả cho "kiến trúc", hãy quy cho "encoder lớp PhoBERT".

**Vì sao ECE thấp hơn ở cascade nhưng không đáng tin?** Xác suất của cascade là **tích hai marginal** nên bị under-confident một cách cấu trúc; ECE 5,3% không có nghĩa nó "calibrated tốt hơn". Đã ghi rõ.

---

## 8. Những gì đã sửa so với bản nháp (rất có thể bị hỏi)

| Lỗi bản nháp | Cách xử lý |
|---|---|
| Baseline và multi-task chạy 2 protocol | một script, một protocol duy nhất |
| Hai con số chính không có file kết quả | mỗi run lưu config, log, metrics, dự đoán từng dòng |
| PDF nộp là bản cũ 1 tác giả, tex là LNCS không build được | chuyển IEEEtran + XeLaTeX, build được, 3 tác giả |
| Trích dẫn sai tác giả bài gốc ("Tran et al.") | sửa thành Dinh & Luu |
| 1 seed, kết luận +0,73 điểm | 3 seed + bootstrap + McNemar → kết luận **không khác biệt** |
| Bảng ablation trộn 2 protocol | chạy lại ablation dưới protocol khoá |

**Nếu bị hỏi "vậy con số cũ sai à?"** Trả lời: "Con số cũ là kết quả của một seed và hai protocol khác nhau; khi khoá protocol và chạy 3 seed thì hiệu ứng biến mất. Nhóm báo cáo lại, không giữ kết luận cũ."

---

## 9. Hạn chế — nói trước, đừng để bị bắt

1. **Spam-1 chỉ 54 dòng test**, dev 17 dòng → F1 của lớp này bất định, và là nguồn nhiễu dừng sớm.
2. **Split disjoint mới là một cách chia** (seed 42/43). Đang chạy thêm 2 partition để đo phương sai giữa các cách chia.
3. **Fusion không phải S-PhoBERT hai tầng** — chỉ là tái cài đặt ở mức khái niệm.
4. **Dedupe chỉ bắt trùng tuyệt đối** ở bản chính; bản near-duplicate đang chạy.
5. **Một dataset, một ngôn ngữ**; chưa có kiểm tra trôi dữ liệu theo thời gian.
6. **Không sweep hyperparameter** (chỉ 2 giá trị LR, 2 cửa sổ).

---

## 10. Mười hai câu hỏi phản biện + câu trả lời mẫu

1. **Sao không so SOTA?** → Bài không đua điểm; bài đo cái gì làm đổi điểm. Bản thân việc tái lập đã là đóng góp.
2. **3 seed đủ chưa?** → Đủ để thấy hiệu ứng kiến trúc (0,1) nhỏ hơn nhiễu (0,7–1,0). Đó chính là phát hiện.
3. **Sao PhoBERT mà không XLM-R?** → Có chạy: 74,63, kém 1,47.
4. **Fusion hỏng do cài sai?** → Hai bản (nối text và mã hoá riêng); bản không cắt cụt vẫn không hơn.
5. **Label noise có bằng chứng?** → 9,4% vs 0,39%; là suy luận gián tiếp, đã ghi rõ.
6. **Sao dùng macro-F1?** → Mất cân bằng 73,5%; accuracy vô nghĩa.
7. **McNemar p=0,047 mà bảo không khác?** → Hai câu hỏi khác nhau: McNemar nói *mẫu lỗi* khác, bootstrap nói *điểm trung bình* không khác.
8. **Vì sao dừng sớm theo dev mà không theo test?** → Dùng test để chọn là rò rỉ; dev là tập duy nhất được phép.
9. **Sao không ensemble nhiều seed?** → Nằm ngoài phạm vi; sẽ làm ở bước sau, cùng protocol.
10. **Kết quả âm có giá trị gì?** → Tiết kiệm thời gian cho người sau; và chứng minh kết luận không phải do may mắn.
11. **Sao không xử lý teencode?** → Đang chạy; bản nháp có thử rule nhưng ngoài protocol nên không dùng.
12. **Nếu làm lại, làm gì trước?** → Tăng số dòng Spam-1 (hoặc gộp lớp), và luôn báo CI thay vì một điểm.

---

## 11. Từ điển nhanh (nếu bị hỏi thuật ngữ)

- **Macro-F1:** trung bình F1 của các lớp, mỗi lớp trọng số bằng nhau.
- **GroupShuffleSplit:** chia dữ liệu theo *nhóm* (ở đây là sản phẩm) để không nhóm nào xuất hiện ở hai split.
- **Bootstrap:** lấy mẫu lại có hoàn lại từ tập test để ước lượng khoảng tin cậy của chênh lệch.
- **McNemar:** kiểm định trên các cặp dự đoán *bất đồng* của hai mô hình.
- **ECE:** sai lệch giữa độ tin cậy và độ chính xác thực tế (calibration).
- **Cascade:** mô hình phân cấp — quyết định cha (spam) rồi mới đến con (loại spam).
- **Label noise:** nhãn sai trong dữ liệu huấn luyện.

---

## 12. Checklist trước khi vào phòng

- [ ] Mở sẵn: PDF bài báo, file slide, repo GitHub (3 tab).
- [ ] Thuộc 6 con số ở mục 1 và 3 con số ở mục 6.
- [ ] Biết mình sẽ nói câu chốt: *"Câu hỏi không phải mô hình nào tốt hơn, mà là đo bằng thước nào."*
- [ ] Nếu quên số: nói "nhóm có bảng chi tiết trong bài, mời xem Bảng X" rồi chuyển.
