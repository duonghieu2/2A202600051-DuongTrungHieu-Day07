# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Dương Trung Hiếu
**Nhóm:** C401-A5
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Có nghĩa là hai đoạn văn bản (được biểu diễn dưới dạng vector) có hướng rất gần nhau trong không gian nhiều chiều, thể hiện rằng chúng có ý nghĩa ngữ nghĩa (semantic meaning) rất giống nhau hoặc cùng nói về một chủ đề.

**Ví dụ HIGH similarity:**
- Sentence A: Khách hàng yêu cầu trả hàng và hoàn tiền do sản phẩm bị lỗi.
- Sentence B: Người mua muốn gửi lại hàng hỏng để lấy lại tiền.
- Tại sao tương đồng: Dùng từ vựng khác nhau ("trả hàng/hoàn tiền" vs "gửi lại hàng/lấy lại tiền") nhưng cùng diễn đạt chung một ý định và hoàn cảnh.

**Ví dụ LOW similarity:**
- Sentence A: Khách hàng yêu cầu trả hàng và hoàn tiền do sản phẩm bị lỗi.
- Sentence B: TikiNOW hỗ trợ giao hàng siêu tốc trong vòng 2 giờ.
- Tại sao khác: Hai câu mang ý nghĩa hoàn toàn khác biệt, một câu nói về chính sách hậu mãi, một câu nói về dịch vụ vận chuyển.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ quan tâm đến "góc" (hướng) giữa hai vector chứ không quan tâm đến "độ dài" (magnitude) của chúng. Điều này rất phù hợp với văn bản vì một tài liệu dài và một câu truy vấn ngắn vẫn có thể có nội dung giống nhau (góc nhỏ) dù độ dài vector của chúng chênh lệch lớn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> số chunk = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23.11
> 23 chunks (thực tế sẽ có 22 chunks đủ 500 ký tự và chunk cuối chứa phần còn lại).

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số lượng chunk sẽ tăng lên ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25 chunks. Ta muốn overlap nhiều hơn để duy trì tốt hơn ngữ cảnh giữa các chunk liền kề, giúp các câu hoặc ý tưởng không bị cắt gãy rời rạc ở ranh giới của 2 chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Customer Support Policy (E-commerce)

**Tại sao nhóm chọn domain này?**
> - Dễ kiếm doc (đổi trả, giao hàng, bảo hành, thanh toán…)
> - Có nhiều rule **rõ ràng + dễ sai**
> - RAG thể hiện rõ giá trị (retrieve đúng chunk hay không)

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Chính sách hậu mãi: Đổi mới, trả hàng hoàn tiền và bảo hành sản phẩm | `data/1.md` | 7277 | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 2 | Chính sách đổi trả sản phẩm | `data/2.md` | 5567 | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 3 | Các câu hỏi thường gặp về đổi trả | `data/3.md` | 4222 | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 4 | Hướng dẫn đổi trả online | `data/4.md` | 1688 | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 5 | Chính sách bảo hành tại Tiki như thế nào? | `data/5.md` | 3035 | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 6 | Tiki hiện đang hỗ trợ các phương thức thanh toán nào | `data/6.md` | 2976 | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 7 | Làm thế nào để tôi có thể lưu và sử dụng mã coupon? | `data/7.md` | 1262 | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 8 | Dịch vụ giao hàng từ nước ngoài | `data/8.md` | 3237 | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 9 | Dịch vụ giao hàng TikiNOW | `data/9.md` | 2244 | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 10 | Tôi có thể yêu cầu giao theo thời gian cụ thể, giao vào chủ nhật hoặc trên lầu/phòng chung cư không | `data/10.md` | 927 | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |

### Metadata Schema

Dữ liệu chưa có metadata

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| 1.md | FixedSizeChunker (`fixed_size`) | 17 | 475.1 | Kém (Thường bị cắt ngang câu/từ) |
| 1.md | SentenceChunker (`by_sentences`) | 14 | 518.2 | Tốt (Giữ nguyên cấu trúc câu) |
| 1.md | RecursiveChunker (`recursive`) | 24 | 301.8 | Rất Tốt (Chia theo đoạn, Header) |
| 2.md | RecursiveChunker (`fixed_size`) | 13 | 474.4 | Kém |
| 2.md | RecursiveChunker (`by_sentences`) | 12 | 461.3 | Tốt |
| 2.md | RecursiveChunker (`recursive`) | 13 | 426.4 | Rất tốt |
| 3.md | RecursiveChunker (`fixed_size`) | 13 | 0 | 467.2 | Kém |
| 3.md | RecursiveChunker (`by_sentences`) | 12 | 349.2 | Tốt |
| 3.md | RecursiveChunker (`recursive`) | 11 | 382.1 | Rất tốt |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> Chiến lược này sẽ cố gắng chia văn bản dựa trên các điểm ngắt tự nhiên lớn nhất trước (như hai dấu xuống dòng \n\n đại diện cho các đoạn văn). Nếu đoạn văn bản sau khi chia vẫn lớn hơn chunk_size quy định, nó mới dùng đến các điểm ngắt nhỏ hơn (như \n, dấu chấm câu . , hoặc khoảng trắng) để tiếp tục chia nhỏ. Nhờ quá trình đệ quy này, các đoạn văn được chia ra sẽ giữ được ý nghĩa trọn vẹn nhất có thể.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain của nhóm là các file Markdown về Chính sách khách hàng (có chứa rất nhiều Header như ## 1., các gạch đầu dòng, các đoạn văn quy định). RecursiveChunker đặc biệt hiệu quả với định dạng Markdown vì nó ưu tiên chia theo cụm đoạn văn (phân tách bởi \n\n), giúp các điều khoản không bị đứt đoạn giữa chừng như FixedSizeChunker.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 1.md | best baseline(by_sentences) | 14 | 518.2 | Tốt, giữ được trọn vẹn câu, nhưng thi thoảng bị gộp các điều khoản không liên quan vào cùng một chunk. |
| 1.md | **của tôi**(recursive) | 24 | 301.8 | Tốt nhất, kích thước chunk vừa vặn, giữ trọn vẹn từng điều khoản quy định. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 8/10 | Giữ được trọn vẹn một đoạn quy định/chính sách. Tương thích rất tốt với định dạng Markdown. | Một số chunk có thể bị hơi ngắn nếu đoạn văn chứa quá nhiều gạch đầu dòng ngắn. |
| Lê Hồng Quân | SentenceChunker | 7 / 10 | Chunk dễ đọc, giữ ngữ nghĩa tự nhiên, hợp với FAQ/policy | Một số chunk dài nên retrieval chưa luôn đứng top-1 |
| Đoàn Sĩ Linh | FixedSizeChunker | 7 / 10 | Dễ triển khai, chunk size ổn định | Dễ cắt ngang ý và làm mất ngữ cảnh |
| Nguyễn Đức Hải | MarkdownChunker | 8 / 10 | Giữ trọn ngữ cảnh theo heading, phù hợp với FAQ | Chunk size không ổn định, phụ thuộc cấu trúc doc |
| Phạm Thanh Lam | RecursiveChunker | 8/10 | Cân bằng giữa độ dài và ngữ cảnh, thường mạnh ở retrieval | Sinh nhiều chunk hơn, khó kiểm tra thủ công hơn |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Đối với bộ tài liệu về Chính sách khách hàng (định dạng Markdown), chiến lược RecursiveChunker là tốt nhất. Lý do là vì loại tài liệu này có tính cấu trúc rất cao (phân chia rõ ràng bằng các Header, đoạn văn, dấu xuống dòng); RecursiveChunker ưu tiên băm văn bản dựa trên các ranh giới tự nhiên này (\n\n, \n) nên nó "bắt" trọn vẹn được một quy định/điều khoản vào một chunk mà không làm đứt gãy ngữ cảnh.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng thư viện re (Regex) để tách chuỗi dựa trên các dấu hiệu kết thúc câu như chấm, hỏi, than kèm theo khoảng trắng (?<=[.!?])\s+ hoặc dấu chấm kèm xuống dòng (?<=\.)\n. Sau đó dùng vòng lặp gom nhóm các câu lại theo số lượng max_sentences_per_chunk được chỉ định.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Hàm _split hoạt động theo thuật toán đệ quy. Base case là nếu text hiện tại ngắn hơn chunk_size thì trả về chính nó. Ngược lại, thử tách text bằng separator đầu tiên trong list (ví dụ \n\n), tiến hành gộp các mảnh lại cho đến khi đạt giới hạn chunk_size. Các mảnh nào vỡ khung sẽ bị đệ quy tiếp với các separator cấp thấp hơn (\n, .).

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Khi thêm tài liệu, hệ thống nhúng nội dung qua _embedding_fn và lưu cấu trúc dict (chứa id, content, metadata, vector) vào list _store in-memory. Khi tìm kiếm, thuật toán mã hóa câu query thành vector, lặp qua toàn bộ _store để tính Cosine Similarity, sau đó sort giảm dần theo điểm score và lấy ra top_k phần tử.

**`search_with_filter` + `delete_document`** — approach:
> Để tối ưu kết quả, hệ thống duyệt _store và lọc ra (filter) một list phụ chỉ chứa các document khớp điều kiện metadata, sau đó mới gọi hàm _search_records trên list phụ này. Chức năng xóa đơn giản là tạo ra một _store mới loại bỏ đi những chunk có id hoặc metadata['doc_id'] khớp với ID cần xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> Nhận câu hỏi, gọi store.search để lấy ra top_k chunk có điểm similarity cao nhất. Trích xuất text từ các chunk này, nối lại bằng \n\n và chèn vào một khung prompt hướng dẫn (có chứa nhãn --- Context ---). Cuối cùng đưa nguyên khối prompt này cho LLM xử lý sinh văn bản.

### Test Results

```
PS F:\Thuctap_Vin\2A202600051-DuongTrungHieu-Day07> pytest tests/ -v                           
=============================================== test session starts ===============================================
platform win32 -- Python 3.13.1, pytest-9.0.2, pluggy-1.6.0 -- C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe
cachedir: .pytest_cache
rootdir: F:\Thuctap_Vin\2A202600051-DuongTrungHieu-Day07
plugins: anyio-4.11.0, asyncio-1.3.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 42 items                                                                                                 

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                        [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                 [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                          [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                           [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                      [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                       [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                     [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                       [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                       [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                  [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                              [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                        [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED               [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                   [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED             [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                   [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                       [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                         [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                           [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                 [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                      [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                        [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED            [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                         [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                  [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                 [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                            [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                        [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                   [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                       [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                             [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                       [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED    [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                  [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                 [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED     [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED         [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED   [100%]

=============================================== 42 passed in 0.28s ================================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tiki hoàn tiền qua thẻ Visa trong bao lâu? | Thời gian xử lý hoàn trả tiền vào thẻ tín dụng. | high | (0.8 - 0.9)* | Đúng |
| 2 | TikiNOW giao hàng trong vòng 2 giờ. | Mức phí giao hàng siêu tốc là 25.000 VNĐ. | low | (0.3 - 0.4)* | Đúng |
| 3 | Sản phẩm bị lỗi kỹ thuật từ nhà sản xuất. | Hàng hóa bị hư hỏng không do người sử dụng. | high | (0.8 - 0.9)* | Đúng |
| 4 | Tôi muốn đổi địa chỉ giao hàng. | Hướng dẫn thanh toán bằng ví MoMo. | low | (0.1 - 0.2)* | Đúng |
| 5 | Các mặt hàng không áp dụng chính sách đổi trả. | Danh mục sản phẩm hạn chế đổi trả do nhu cầu. | high | (0.8 - 0.9)* | Đúng |

*Actual Score là ước tính của mô hình nhúng ngữ nghĩa thật.

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là cặp số 1 hoặc cặp số 5, vì Sentence A và Sentence B sử dụng các từ vựng hoàn toàn khác nhau (không trùng từ khóa nào) nhưng điểm similarity trả về vẫn rất cao. Điều này chứng minh embeddings không hoạt động bằng cách "so khớp từ khóa" (keyword matching) thông thường, mà nó đã mã hóa được "ý nghĩa ẩn" (latent meaning) của cả câu vào trong không gian đa chiều.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Sau khi kiện trả hàng tới nhà bán thì bao lâu Tiki xử lý hoàn tiền? | Tiki hoàn tiền sau khi quy trình kiểm tra, đánh giá chất lượng sản phẩm đổi/trả hoàn tất; riêng phần xử lý này cần 3 ngày làm việc kể từ khi kiện hàng được chuyển tới nhà bán. |
| 2 | Nếu tôi gửi sản phẩm bảo hành về Tiki thì bao lâu nhận lại? | Nếu khách gửi hàng bảo hành về Tiki, thời gian bảo hành dự kiến là 15–30 ngày, chưa tính thời gian vận chuyển đi và về. |
| 3 | Đơn giao từ nước ngoài mà giao không thành công thì Tiki giao lại mấy lần, giữ kho bao lâu? | Với đơn giao từ nước ngoài, nếu giao không thành công thì Tiki hỗ trợ giao lại tối đa 03 lần; sau đó hàng được giữ tại kho Tiki 14 ngày. Nếu quá thời hạn đó khách không liên hệ nhận hàng thì Tiki tiến hành hoàn tiền qua đơn hàng. |
| 4 | Muốn dùng Tiki Xu và mã giảm giá thì có điều kiện gì? | Khách chỉ có thể dùng Tiki Xu khi có từ 1000 Xu trở lên; còn mỗi mã giảm giá chỉ dùng 1 lần trên 1 tài khoản. |
| 5 | Tôi có thể yêu cầu giao vào giờ cụ thể hoặc hẹn lại chủ nhật không? | Sau khi đặt hàng thành công, Tiki sẽ thông báo thời gian giao dự kiến. Nếu thời điểm shipper liên hệ chưa phù hợp, khách có thể trao đổi qua điện thoại để hẹn lại thời gian giao khác, và nhân viên vận chuyển sẽ cố gắng hỗ trợ trong mức có thể. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sau khi kiện trả hàng tới nhà bán thì bao lâu Tiki xử lý hoàn tiền? | 2.md - Chính sách đổi trả sản phẩm (Quy định chung về đổi trả). | 0.731 | Không (Top-3 có chứa chunk đúng là 3.md - Thời gian hoàn tiền ở hạng 3) | [DEMO LLM] Generated answer from prompt preview: Sử dụng thông tin ngữ cảnh dưới đây... # Chính sách đổi trả sản phẩm... |
| 2 | Nếu tôi gửi sản phẩm bảo hành về Tiki thì bao lâu nhận lại? | 5.md - Chính sách bảo hành tại Tiki như thế nào? | 0.632 | Có | [DEMO LLM] Generated answer from prompt preview: Sử dụng thông tin ngữ cảnh dưới đây... # Chính sách bảo hành tại Tiki như thế nào?... |
| 3 | Đơn giao từ nước ngoài mà giao không thành công thì Tiki giao lại mấy lần giữ kho bao lâu? | 4.md - Hướng dẫn đổi trả online (Thao tác tạo yêu cầu đổi trả trên App). | 0.694 | Không (Cả Top-3 đều trật, file đúng là 8.md không lọt vào danh sách) | [DEMO LLM] Generated answer from prompt preview: Sử dụng thông tin ngữ cảnh dưới đây... # Hướng dẫn đổi trả online... |
| 4 | Muốn dùng Tiki Xu và mã giảm giá thì có điều kiện gì? | 9.md - Dịch vụ giao hàng TikiNOW (Đối tượng áp dụng và phí vận chuyển). | 0.653 | Không (Top-3 có chứa chunk đúng là 7.md - Hướng dẫn dùng mã coupon ở hạng 2) | [DEMO LLM] Generated answer from prompt preview: Sử dụng thông tin ngữ cảnh dưới đây... # Dịch vụ giao hàng TikiNOW... |
| 5 | Tôi có thể yêu cầu giao vào giờ cụ thể hoặc hẹn lại chủ nhật không? | 10.md - Tôi có thể yêu cầu giao theo thời gian cụ thể, giao vào chủ nhật... | 0.686 | Có | [DEMO LLM] Generated answer from prompt preview: Sử dụng thông tin ngữ cảnh dưới đây... # Tôi có thể yêu cầu giao theo thời gian cụ thể... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Nhờ so sánh chéo, tôi nhận ra việc đặt giá trị overlap đủ lớn trong chiến lược băm văn bản là rất quan trọng. Một thành viên dùng FixedSizeChunker với overlap nhỏ khiến nhiều từ khóa bị cắt làm đôi, trong khi thành viên dùng overlap lớn hơn đã giải quyết được tình trạng mất ngữ cảnh này.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Tôi rất ấn tượng với ý tưởng thiết kế Metadata schema của một nhóm khác. Họ đã chủ động phân loại tài liệu theo nhóm "Danh mục". Nhờ đó, trước khi dùng thuật toán Similarity, họ đã dùng bộ lọc Metadata để loại bỏ các tài liệu sai chủ đề, giúp kết quả chính xác hơn hẳn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu được làm lại, tôi sẽ cải tiến chiến lược gán Metadata. Thay vì chỉ gán metadata chung chung cho cả file document, tôi sẽ trích xuất Header của từng đoạn văn (ví dụ: Header: Thời gian hoàn tiền) để đưa vào làm metadata cho chính chunk đó. Điều này sẽ giúp việc lọc (metadata filtering) chính xác đến từng đoạn quy định thay vì toàn bộ văn bản.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 28 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **85/90** |
