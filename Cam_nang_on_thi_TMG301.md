# Cẩm nang ôn kiểm tra TMG301 - Text Mining

## 0. Bản đồ toàn môn

```text
Raw text
  ↓
Regular expression + Text normalization
  ↓
Tokenization
  ↓
N-gram / TF-IDF / Embedding
  ↓
Classification hoặc Similarity
  ↓
Clustering
  ↓
Evaluation
```

- **Regex:** tìm mẫu trong văn bản.
- **Normalization:** đưa các cách viết về cùng chuẩn.
- **N-gram:** mô hình hóa từ đơn hoặc chuỗi từ.
- **Classification:** dự đoán văn bản thuộc nhãn nào.
- **Embedding:** biểu diễn ý nghĩa bằng vector.
- **Clustering:** tự chia văn bản thành các nhóm.

---

## 1. Regular Expressions, Text Normalization và Edit Distance

### 1.1. Regex

Regex là mẫu ký tự dùng để tìm, kiểm tra hoặc trích xuất chuỗi.

| Ký hiệu | Ý nghĩa |
|---|---|
| `.` | Một ký tự bất kỳ |
| `\\d` | Chữ số, tương đương `[0-9]` |
| `\\w` | Chữ, số hoặc `_` |
| `\\s` | Khoảng trắng |
| `+` | Lặp ít nhất một lần |
| `*` | Lặp không hoặc nhiều lần |
| `?` | Không hoặc một lần |
| `{m,n}` | Từ m đến n lần |
| `[]` | Một ký tự thuộc tập |
| `()` | Nhóm biểu thức |
| `^` | Bắt đầu chuỗi |
| `$` | Kết thúc chuỗi |
| `\\b` | Ranh giới ký tự từ |

Ví dụ ngày `YYYY-MM-DD`:

```python
re.findall(r'\\b\\d{4}-\\d{2}-\\d{2}\\b', text)
```

Email cơ bản:

```regex
\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b
```

`\\b` giúp tránh lấy một phần của chuỗi dài hơn. Regex này phù hợp bài tập cơ bản, chưa phải bộ kiểm tra email đầy đủ theo RFC.

### 1.2. Các hàm `re`

- `re.match()`: chỉ kiểm tra từ đầu chuỗi.
- `re.search()`: tìm lần đầu ở bất kỳ vị trí nào.
- `re.findall()`: trả về danh sách mọi kết quả.
- `re.finditer()`: trả về match object và vị trí.

### 1.3. Normalization

```python
text = text.lower()
text = re.sub(r'[^a-z0-9\\s]', ' ', text)
text = re.sub(r'\\s+', ' ', text).strip()
```

Pipeline thường gặp:

1. Chuẩn hóa Unicode.
2. Lowercase nếu không cần phân biệt hoa thường.
3. Tokenization.
4. Xóa ký tự không cần thiết.
5. Xóa stopwords nếu phù hợp.
6. Stemming hoặc lemmatization.

Với tiếng Việt, không nên xóa máy móc mọi ký tự ngoài ASCII vì dấu tiếng Việt mang thông tin.

### 1.4. Edit distance

Levenshtein distance là số thao tác tối thiểu để biến chuỗi A thành B:

- insert: thêm ký tự;
- delete: xóa ký tự;
- substitute: thay ký tự.

```text
D[i,j] = min(
    D[i-1,j] + 1,
    D[i,j-1] + 1,
    D[i-1,j-1] + cost
)
```

`cost = 0` nếu hai ký tự giống nhau, ngược lại bằng 1.

Mẹo nhớ: **đi xuống là xóa, đi ngang là thêm, đi chéo là đổi hoặc giữ**.

### Lỗi dễ mất điểm

- Quên escape dấu chấm: `.` nghĩa là mọi ký tự; `\\.` mới là dấu chấm thật.
- Dùng `re.match()` khi đề yêu cầu tìm toàn văn bản.
- Xóa dấu tiếng Việt trong normalization.
- Nhầm edit distance với số vị trí khác nhau giữa hai chuỗi.

---

## 2. N-gram Language Models

### 2.1. Ý tưởng

Với câu `I love NLP`:

```text
Unigram: I, love, NLP
Bigram: (I, love), (love, NLP)
```

- **Unigram:** xác suất một từ, không phụ thuộc từ trước.
- **Bigram:** xác suất từ hiện tại phụ thuộc một từ trước.
- **Trigram:** phụ thuộc hai từ trước.

### 2.2. Công thức

```text
P(w) = count(w) / total_tokens

P(w_i | w_{i-1}) =
count(w_{i-1}, w_i) / count(w_{i-1})
```

Xác suất câu theo bigram:

```text
P(w1,...,wn) ≈ Π P(w_i | w_{i-1})
```

Nên thêm token biên:

```text
<s> I love NLP </s>
```

### 2.3. Smoothing

Nếu bigram chưa xuất hiện, xác suất bằng 0 và kéo cả xác suất câu về 0. Add-one smoothing:

```text
P(w_i | w_{i-1}) =
(count(w_{i-1},w_i)+1) /
(count(w_{i-1}) + V)
```

`V` là vocabulary size.

- Unigram sampling chọn từ theo tần suất chung, nên ít giữ ngữ pháp.
- Bigram Markov chain chọn từ theo từ hiện tại, nên liên kết cục bộ tốt hơn.

Mẹo nhớ: **Unigram biết từ nào phổ biến; Bigram biết từ nào đi sau từ nào**.

### Lỗi dễ mất điểm

- Chia bigram cho tổng token thay vì count của từ điều kiện.
- Nhầm `P(a|b)` với `P(b|a)`.
- Quên smoothing hoặc token biên.

---

## 3. Naive Bayes cho Text Classification và Sentiment

### 3.1. Ý tưởng cốt lõi

Naive Bayes chọn lớp có posterior lớn nhất:

```text
P(class | document) ∝ P(class) × P(document | class)
```

Với Multinomial Naive Bayes:

```text
P(c | d) ∝ P(c) × Π P(word | c)^{count(word,d)}
```

Giả định “naive” là các từ độc lập có điều kiện khi đã biết lớp. Giả định này không hoàn toàn đúng nhưng model thường rất nhanh và hữu ích cho text classification.

### 3.2. Quy trình

```text
Text → preprocessing → CountVectorizer/TF-IDF
     → prior + likelihood → Naive Bayes → label
```

Laplace smoothing:

```text
P(word|class) = (count(word,class)+α) /
(total_words_in_class + αV)
```

Smoothing ngăn xác suất bằng 0 khi một từ chưa xuất hiện trong lớp.

### 3.3. Sentiment

Model học phân phối từ của lớp tích cực và tiêu cực. Ví dụ `excellent`, `tốt`, `great` có thể nghiêng về positive; `slow`, `bad`, `tệ` có thể nghiêng về negative.

Review như “Không tệ” có thể gây khó vì model có thể xử lý riêng `không` và `tệ`, thay vì hiểu toàn bộ phủ định.

### Ưu và nhược điểm

- Ưu: nhanh, dễ cài đặt, hoạt động tốt với vector sparse.
- Nhược: giả định độc lập, khó hiểu thứ tự từ, phủ định và quan hệ xa.

### Lỗi dễ mất điểm

- Fit vectorizer trên cả train và test, gây data leakage.
- Không dùng cùng vectorizer cho train và test.
- Quên smoothing.
- Chỉ báo cáo accuracy khi dữ liệu mất cân bằng.

---

## 4. Logistic Regression

### 4.1. Ý tưởng

Logistic Regression là mô hình tuyến tính dùng score của feature để dự đoán xác suất:

```text
z = w^T x + b
P(y=1|x) = sigmoid(z) = 1/(1+e^{-z})
```

Nếu xác suất lớn hơn ngưỡng, thường là 0.5, model dự đoán lớp 1.

- `w_j > 0`: feature làm tăng score của lớp 1.
- `w_j < 0`: feature làm giảm score.
- `|w_j|` lớn: feature có ảnh hưởng mạnh hơn nếu scale phù hợp.

### 4.2. Huấn luyện

1. Khởi tạo `w`, `b`.
2. Tính logit `z`.
3. Dùng sigmoid để lấy xác suất.
4. Tính cross-entropy.
5. Gradient descent cập nhật tham số.
6. Lặp đến khi hội tụ.

```text
L = -[y log(p) + (1-y) log(1-p)]
```

### 4.3. Regularization

```text
L2: L_total = L + λ Σ w_j²
L1: L_total = L + λ Σ |w_j|
```

- L1 có thể đưa nhiều trọng số về 0, giống feature selection.
- L2 làm trọng số nhỏ và ổn định hơn.
- Trong scikit-learn, `C` lớn thường nghĩa là regularization yếu hơn.

### Lỗi dễ mất điểm

- Nhầm đây là regression dự đoán số liên tục.
- Quên sigmoid.
- Không chuẩn hóa khi dùng feature có scale rất khác nhau.
- Fit TF-IDF sai quy trình.

---

## 5. Vector Semantics và Embeddings

### 5.1. Vì sao cần vector?

Máy tính không hiểu trực tiếp câu chữ. Ta ánh xạ từ hoặc câu thành vector để tính độ tương tự, phân loại, tìm kiếm hoặc gom cụm.

One-hot có vector thưa và mọi từ khác nhau đều cách nhau như nhau, nên không biểu diễn được quan hệ “tốt” gần “tuyệt”.

### 5.2. Distributional hypothesis

> Những từ xuất hiện trong ngữ cảnh giống nhau thường có ý nghĩa gần nhau.

### 5.3. Word2Vec

- **CBOW:** dùng các từ xung quanh để dự đoán từ trung tâm.
- **Skip-gram:** dùng từ trung tâm để dự đoán các từ xung quanh.

CBOW thường nhanh hơn; Skip-gram thường hữu ích hơn với từ hiếm.

### 5.4. Cosine similarity

```text
cos(a,b) = (a·b) / (||a|| ||b||)
```

- gần 1: cùng hướng, tương tự cao;
- gần 0: ít liên quan;
- gần -1: hướng đối lập nếu không gian cho phép.

Cosine đo hướng của vector, không chỉ đo độ dài.

### 5.5. Sentence embedding

Có thể lấy trung bình word vectors, nhưng cách đó mất thứ tự và ngữ cảnh. Transformer hoặc SBERT tạo sentence embedding giàu ngữ nghĩa hơn.

Mẹo nhớ: **TF-IDF biết từ quan trọng trong tài liệu; embedding biết quan hệ ý nghĩa**.

### Lỗi dễ mất điểm

- Nghĩ hai vector gần nhau chỉ vì độ dài lớn.
- Nhầm cosine similarity với cosine distance.
- So sánh vector từ hai không gian embedding khác nhau.
- Quên static embedding thường không phân biệt nghĩa theo từng ngữ cảnh.

---

## 6. Clustering for Text Similarity

### 6.1. Supervised và unsupervised

- **Supervised:** có nhãn, ví dụ Naive Bayes và Logistic Regression.
- **Unsupervised:** không có nhãn, model tự tìm nhóm, ví dụ K-Means.

### 6.2. Pipeline K-Means

```text
Documents → preprocessing → TF-IDF/embedding
          → chọn K centroid → gán điểm vào centroid gần nhất
          → cập nhật centroid → lặp đến ổn định
```

K-Means tối thiểu hóa tổng khoảng cách trong cụm:

```text
SSE = Σ_k Σ_(x_i in C_k) ||x_i - μ_k||²
```

### 6.3. Chọn K

- **Elbow:** vẽ SSE theo K, chọn điểm mà thêm cụm không còn giảm SSE đáng kể.
- **Silhouette:** đo điểm có gần cụm của nó và xa cụm khác đến đâu.

Silhouette gần 1 là tốt; gần 0 là các cụm chồng lấn; âm là có thể gán sai.

### 6.4. Với văn bản

TF-IDF tạo vector sparse. Cosine similarity thường phù hợp để đo văn bản, nhưng phải nhất quán giữa vector hóa và metric.

### Lỗi dễ mất điểm

- Cluster ID 0, 1, 2 không tự có ý nghĩa cố định.
- Không đánh giá clustering bằng accuracy nếu không có nhãn chuẩn.
- Chọn K tùy tiện.
- Diễn giải centroid như một document thật.

---

## 7. Bảng so sánh nhanh

| Phương pháp | Có nhãn? | Đầu vào | Mạnh ở đâu? | Yếu ở đâu? |
|---|---:|---|---|---|
| Regex | Không | Chuỗi thô | Trích xuất mẫu rõ ràng | Không hiểu ngữ nghĩa |
| N-gram | Không/LM | Token | Xác suất cục bộ | Sparse, phụ thuộc dữ liệu |
| Naive Bayes | Có | Count/TF-IDF | Nhanh, dễ làm baseline | Giả định độc lập |
| Logistic Regression | Có | TF-IDF | Biên tuyến tính, dễ giải thích | Không hiểu ngữ cảnh sâu |
| Embedding | Có thể có/không | Từ/câu | Độ tương tự ngữ nghĩa | Phụ thuộc dữ liệu học vector |
| K-Means | Không | Vector | Gom nhóm nhanh | Cần chọn K, nhạy outlier |

---

## 8. Công thức cần học thuộc

```text
Unigram:
P(w) = count(w) / total_tokens

Bigram:
P(w_i|w_{i-1}) = count(w_{i-1},w_i) / count(w_{i-1})

Naive Bayes:
P(c|d) ∝ P(c) Π P(w|c)^{count(w,d)}

Sigmoid:
σ(z) = 1 / (1 + e^{-z})

Logistic logit:
z = w^T x + b

Cosine:
cos(a,b) = (a·b)/(||a||||b||)

K-Means:
SSE = Σ ||x_i - μ_cluster(i)||²
```

---

## 9. Mẫu pipeline coding

### Classification

```python
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer()
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

model.fit(Xtr, y_train)
pred = model.predict(Xte)
```

Nhớ: chỉ `fit_transform` trên train, test chỉ `transform`.

### Evaluation

```python
accuracy_score(y_test, pred)
classification_report(y_test, pred)
confusion_matrix(y_test, pred)
```

Khi dữ liệu lệch lớp, nhìn thêm precision, recall, macro-F1 và confusion matrix.

### Clustering

```python
X = TfidfVectorizer().fit_transform(documents)
model = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_id = model.fit_predict(X)
```

---

## 10. Mẹo nhớ nhanh

- **Regex tìm hình dạng, không hiểu nghĩa.**
- **Normalization đưa các cách viết về cùng chuẩn.**
- **Unigram nhìn một từ; bigram nhìn cặp từ.**
- **Naive Bayes nhân xác suất; Logistic Regression dùng sigmoid.**
- **TF-IDF đo độ quan trọng của từ; embedding đo quan hệ ngữ nghĩa.**
- **Classification có nhãn; clustering tự tìm nhóm.**
- **Accuracy nhìn tổng thể; Macro-F1 bảo vệ lớp thiểu số.**
- **Fit train, transform test. Không fit test.**
- **Có xác suất bằng 0 thì nghĩ đến smoothing.**
- **Có vector thì nghĩ đến cosine hoặc khoảng cách.**

---

## 11. Câu hỏi tự kiểm tra

1. Vì sao regex ngày cần `\\d{4}-\\d{2}-\\d{2}`?
2. Vì sao `.` phải escape khi muốn tìm dấu chấm?
3. Tại sao bigram cần smoothing?
4. Vì sao Naive Bayes gọi là “naive”?
5. Sigmoid biến logit thành xác suất như thế nào?
6. L1 và L2 regularization khác nhau ra sao?
7. Vì sao TF-IDF không giống embedding?
8. Cosine similarity đo cái gì?
9. K-Means cần K trước vì sao?
10. Khi dữ liệu lệch lớp, vì sao Macro-F1 quan trọng?
11. Tại sao không được fit vectorizer trên test?
12. Unigram và bigram khác nhau ở điều kiện xác suất nào?

Nếu trả lời được 12 câu này bằng lời của mình và tự viết được pipeline classification/clustering, bạn đã nắm phần lõi của 6 bài.

---

## 12. Checklist trước khi thi

- [ ] Viết được regex email, ngày, số điện thoại.
- [ ] Giải thích được `\\b`, `\\d`, `+`, `*`, `?`, `{m,n}`.
- [ ] Tính được unigram và bigram probability.
- [ ] Biết vì sao cần `<s>`, `</s>` và smoothing.
- [ ] Tính được posterior Naive Bayes ở mức đơn giản.
- [ ] Vẽ được sigmoid và giải thích logit.
- [ ] Phân biệt L1 và L2.
- [ ] Tính cosine similarity bằng tích vô hướng.
- [ ] Giải thích CBOW và Skip-gram.
- [ ] Vẽ được các bước K-Means.
- [ ] Biết chọn metric cho classification và clustering.
- [ ] Nhớ nguyên tắc không gây data leakage.

### Câu chốt toàn môn

> Làm sạch văn bản đúng, biến văn bản thành biểu diễn phù hợp, chọn thuật toán đúng với mục tiêu, rồi đánh giá bằng metric phù hợp.


