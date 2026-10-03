# Báo cáo kết quả — Day 17: Memory Systems for AI Agent

Báo cáo so sánh **Baseline Agent** (chỉ short-term memory trong thread) với **Advanced Agent** (short-term + `User.md` + compact memory) trên hai bộ dữ liệu tiếng Việt trong `data/`. Mọi số liệu dưới đây là từ chế độ **offline** (deterministic, lặp lại được), ngưỡng compact mặc định **800 tokens**, giữ lại **4 message** gần nhất.

Tái hiện:

```bash
python src/benchmark.py          # 2 bảng benchmark
pytest src/test_agents.py -v     # 12 test
```

## 1. Kết quả benchmark

### Standard Benchmark — `conversations.json` (10 hội thoại, 101 lượt, 14 câu hỏi recall)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline | 3,147 | 17,904 | 0%   | 0.15 | 0   | 0 |
| Advanced | 3,282 | 28,054 | 100% | 1.00 | 321 | 0 |

Advanced so với Baseline: prompt tokens **+56,7%**, agent tokens +4,3%, recall +100 điểm %.

### Long-Context Stress Benchmark — `advanced_long_context.json` (1 hội thoại, 16 lượt rất dài, 3 câu hỏi recall)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline | 2,723 | 23,334 | 0%   | 0.15 | 0   | 0 |
| Advanced | 2,770 | 11,059 | 100% | 1.00 | 256 | 5 |

Advanced so với Baseline: prompt tokens **−52,6%**, agent tokens +1,7%, recall +100 điểm %.

## 2. Ba lớp memory được tách bạch thế nào

| Lớp | Ở đâu trong code | Lưu gì | Sống bao lâu |
|---|---|---|---|
| Short-term | `BaselineAgent.sessions`, `CompactMemoryManager.state[thread]["messages"]` | Nguyên văn các message gần nhất | Trong 1 thread |
| Persistent | `UserProfileStore` → `state/profiles/<user>/User.md` | Fact ổn định: tên, nơi ở, nghề, style, đồ uống, món ăn, thú cưng, mối quan tâm | Qua mọi thread và cả khi khởi động lại process |
| Compact | `CompactMemoryManager.state[thread]["summary"]` | Tóm tắt có giới hạn (≤ 6 bullet) các message cũ, ưu tiên câu chứa fact | Trong 1 thread, thay thế phần lịch sử cũ |

Quy tắc phân loại: chỉ **câu khẳng định về bản thân người dùng** mới vào `User.md`. Câu hỏi, yêu cầu recall ("Nhắc lại giúp mình…"), câu đùa ("product manager"), nơi đi họp ("Hà Nội") và nội dung tin tức đều không được lưu. Tin tức và ngữ cảnh tạm thời chỉ nằm ở short-term, sau đó bị nén vào summary.

## 3. Vì sao Advanced có recall tốt hơn Baseline

Câu hỏi recall luôn được hỏi ở **thread mới**. Baseline chỉ có `sessions[thread_id]`, nên sang thread mới nó không còn gì để trả lời. Đây đúng là hành vi mong muốn của baseline, không phải lỗi. Trong cùng thread, Baseline vẫn trả lời đúng (`test_cross_session_recall` kiểm chứng cả hai chiều).

Advanced đọc `User.md` ở mọi thread. Hai agent dùng **chung** một hàm ghép câu trả lời (`answer_from_facts`) và chung extractor, nên chênh lệch recall 0% → 100% chỉ đến từ việc có hay không có persistent memory.

Phần khó nhất là **correction**, không phải ghi nhớ. `upsert_fact` **ghi đè** fact cũ thay vì ghi thêm, nên `User.md` không bao giờ chứa đồng thời "Đà Nẵng" và "Huế":

- Standard: Đà Nẵng → **Huế** (conv-03), backend → **MLOps engineer** (conv-06). Câu "nhắc lại Đà Nẵng như ví dụ cũ" ở conv-10 bị bỏ qua.
- Stress: Huế → **Đà Nẵng** (lượt 9, nằm cùng một câu với "Huế"). Extractor lấy nơi ở được nhắc **sau cùng**. Các mention bị phủ định ("không còn làm backend engineer", "đừng nói backend engineer") bị loại.

## 4. Vì sao Advanced tốn hơn ở hội thoại ngắn

Ở bộ Standard, mỗi hội thoại chỉ khoảng 300 tokens, thấp hơn nhiều so với ngưỡng 800, nên **compact không kích hoạt lần nào**. Khi đó Advanced gửi đúng những gì Baseline gửi, cộng thêm:

- `User.md`: khoảng 57–80 tokens, tăng dần qua 10 hội thoại;
- system prompt dài hơn (có hướng dẫn dùng hồ sơ).

Phần chi phí thêm khoảng 90 tokens/lượt × 115 lượt (101 lượt hội thoại + 14 câu hỏi) ≈ 10.150 tokens. Con số này khớp với chênh lệch 28.054 − 17.904. Agent tokens gần như không đổi (+4,3%) vì câu trả lời recall của Advanced dài hơn câu "chưa có thông tin" của Baseline.

Kết luận: với hội thoại ngắn, persistent memory là **chi phí cố định mỗi lượt** chưa được bù lại bằng gì về token. Thứ nhận lại là recall, không phải tiết kiệm. `test_advanced_costs_more_on_short_threads` ghi nhận trade-off này bằng test.

## 5. Vì sao compact giúp Advanced thắng ở hội thoại dài

### Prompt mỗi lượt trên bộ stress

| Lượt | 1 | 4 | 5 | 8 | 10 | 13 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 220 | 736 | 909 | 1,405 | 1,686 | 2,164 | 2,612 |
| Advanced | 279 | 795 | **556** | 722 | **550** | **600** | 712 |
| Số lần compact (cộng dồn) | 0 | 0 | 1 | 2 | 3 | 4 | 5 |

- Baseline gửi lại **toàn bộ lịch sử** mỗi lượt. Prompt mỗi lượt tăng tuyến tính, nên tổng prompt tăng theo **bình phương** số lượt.
- Advanced đắt hơn ở 4 lượt đầu (thêm User.md), rồi mỗi lần vượt 800 tokens thì nén về khoảng 550–900 tokens. Gồm system prompt, `User.md` (57 tokens), summary (153 tokens) và 4 message gần nhất. Prompt mỗi lượt **dao động trong một dải cố định**, không tăng theo độ dài hội thoại.

### Độ nhạy theo ngưỡng compact (Advanced; Baseline = 23.334 ở bộ stress, 17.904 ở bộ standard)

| Ngưỡng (tokens) | Prompt bộ stress | Compactions bộ stress | Prompt bộ standard | Compactions bộ standard | Recall |
|---:|---:|---:|---:|---:|---:|
| 300  | 9,004  | 28 | 27,782 | 2 | 100% |
| 500  | 9,004  | 17 | 28,054 | 0 | 100% |
| **800** | **11,059** | **5** | 28,054 | 0 | 100% |
| 1200 | 14,262 | 2  | 28,054 | 0 | 100% |
| 2000 | 18,129 | 1  | 28,054 | 0 | 100% |
| 4000 | 24,697 | 0  | 28,054 | 0 | 100% |

Bảng này cho thấy ba điều:

1. **Mức tiết kiệm đến từ compact, không phải từ `User.md`.** Ở ngưỡng 4000 không có lần compact nào, và Advanced **tốn hơn** Baseline (24.697 so với 23.334).
2. **Có điểm bão hoà.** Dưới khoảng 500 tokens, prompt không giảm thêm nữa: 4 message gần nhất của bộ stress đã chiếm khoảng 600 tokens. Lúc này compact chạy ở gần như mọi lượt (28 lần) mà không còn lợi gì. Với LLM summarizer thật, mỗi lần compact là **một lần gọi model**, nên ngưỡng quá thấp sẽ phản tác dụng.
3. **Compact tối ưu `Prompt tokens processed`, không tối ưu `Agent tokens only`.** Agent tokens (lời người dùng + lời agent) gần như bằng nhau giữa hai agent (2.723 so với 2.770), vì compact không làm người dùng nói ít hơn hay agent trả lời ngắn hơn. Thứ compact cắt bớt là phần **ngữ cảnh bị gửi lại** mỗi lượt. Đây cũng là phần chiếm chủ yếu chi phí input của API.

Ngưỡng 800 được chọn để bộ standard **không** bị compact (giữ nguyên ngữ cảnh khi không cần nén), còn bộ stress bị compact đủ nhiều để thấy rõ tác dụng.

## 6. Memory file tăng trưởng thế nào và rủi ro đi kèm

Kích thước `User.md` sau mỗi hội thoại ở bộ standard (bytes):

| conv | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bytes | 226 | 234 | 254 | 296 | 323 | 321 | 318 | 318 | 321 | 321 |

File tăng nhanh khi gặp loại fact mới (conv-01, 04, 05), rồi **đi ngang** khi chỉ còn correction và nhắc lại. Có hai nguyên nhân: upsert ghi đè chứ không ghi thêm, và `interests` bị giới hạn 8 mục. Cần lưu ý: `User.md` được gửi kèm **mọi lượt**, nên mỗi byte thêm vào sẽ nhân lên theo số lượt của mọi hội thoại sau đó.

Rủi ro:

- **Lưu sai fact.** Một mẫu regex sai sẽ ghi một fact sai vào bộ nhớ *vĩnh viễn* và lặp lại nó ở mọi phiên. Ví dụ đã gặp và chặn được: "corgi tên Bơ" suýt bị nhận là tên người dùng, câu đùa "product manager" suýt thành nghề. Lỗi ở persistent memory nguy hiểm hơn lỗi ở short-term vì nó không tự mất đi.
- **Fact cũ không được đính chính.** Ghi đè chỉ đúng khi người dùng *có* đính chính. Một fact đúng hôm nay (ví dụ "đang ở Đà Nẵng vài tháng") sẽ thành sai sau vài tháng mà không có tín hiệu nào. Memory decay ở mục 8 giảm nhẹ rủi ro này.
- **Danh sách phình to.** `interests` và `style` là trường gộp dồn. Nếu không giới hạn, chúng tăng theo số phiên và làm tăng chi phí mỗi lượt.
- **Summary làm mất chi tiết.** Summary heuristic giữ câu chứa fact nhưng cắt bớt phần còn lại. Ví dụ, sau khi compact, các con số trong tin tức (Mach 1.1, 80% El Niño) không còn ở dạng nguyên văn. Với các câu hỏi follow-up về chi tiết cũ trong cùng thread, Advanced sẽ kém hơn Baseline. Đây là cái giá của việc tiết kiệm token.
- **Quyền riêng tư và prompt injection.** `User.md` là file thuần văn bản chứa thông tin cá nhân, và được đưa thẳng vào system prompt. Ở chế độ live, nếu model được phép ghi tự do qua tool `update_user_fact`, người dùng có thể "cấy" chỉ dẫn vào bộ nhớ. Vì vậy code luôn chạy extractor deterministic làm guardrail, và chỉ ghi dưới dạng các dòng `- key: value`.

## 7. Hạn chế của đánh giá này

- **Recall 100% là kết quả lạc quan.** Extractor dùng regex và được tinh chỉnh trên chính bộ dữ liệu này. Với cách diễn đạt mới, độ phủ sẽ thấp hơn. Con số đáng tin hơn là *khoảng cách* giữa hai agent: hai agent dùng chung extractor và chung bộ trả lời, chỉ khác nguồn facts.
- **Response quality ở chế độ offline là heuristic** (70% độ phủ fact, 15% ngắn gọn, 15% không từ chối), nên gần như trùng với recall. Ở chế độ live, benchmark dùng judge model (`JUDGE_*`) để chấm.
- **Token được ước lượng bằng `len/4`.** Tiếng Việt có dấu thường tốn nhiều token hơn mức này với tokenizer thật, nên con số tuyệt đối thấp hơn thực tế. Tỷ lệ so sánh giữa hai agent vẫn có ý nghĩa vì dùng cùng một bộ ước lượng.
- **Chưa đo trên LLM thật.** Ở chế độ live, `SummarizationMiddleware` gọi model để tóm tắt. Khi đó compact có thêm chi phí output và độ trễ, mà benchmark offline chưa phản ánh.

## 8. Bonus: confidence threshold và memory decay

Hai bonus dùng chung một file metadata `User.meta.json`, đặt cạnh `User.md`. File này lưu `confidence`, `mentions` (số lần nhắc), `last_seen` (lượt cuối được nhắc) cho từng fact, danh sách `pending`, và `clock` (bộ đếm lượt của người dùng). Metadata **không** được đưa vào prompt, nên không tốn thêm token mỗi lượt. Bộ standard tạo ra 789 bytes metadata, so với 321 bytes của `User.md`. Cột Memory growth chỉ đo `User.md`, vì đó là phần thật sự được gửi vào prompt.

### 8.1. Confidence threshold (`extract_profile_candidates`, `UserProfileStore.observe`)

**Vấn đề giải quyết.** Phiên bản trước ghi mọi fact khớp regex. Các câu như "Hình như tên mình là Tuấn", "Bạn mình đang ở Hà Nội", "Nếu mình chuyển ra Đà Nẵng…" hay "Tạm thời mình ở Huế" đều bị ghi vào bộ nhớ vĩnh viễn như fact chắc chắn.

**Cách làm.**

- Mỗi fact ứng viên có điểm confidence:
  - điểm gốc theo loại fact: tên 0,95; nghề 0,85; nơi ở 0,8; câu "đồ uống yêu thích là…" 0,9 nhưng "vẫn uống…" chỉ 0,7;
  - trừ điểm theo câu: rào đón hoặc tạm thời −0,4; câu điều kiện "nếu" −0,3 (không áp cho style, vì style thường được nói dạng "khi giải thích, hãy…"); nói về người khác −0,5.
- Chỉ ghi vào `User.md` khi điểm ≥ `MEMORY_MIN_CONFIDENCE` (mặc định 0,6).
- Ứng viên dưới ngưỡng:
  - nếu **trùng** fact đang có, chỉ làm mới `last_seen` (được tính là một lần nhắc lại);
  - nếu **mới**, đưa vào `pending`. Các lần nhắc độc lập được gộp bằng noisy-OR (1 − ∏(1 − cᵢ)). Ví dụ: nói hai lần "Hình như tên mình là Tuấn" (0,55) thì gộp thành 0,80 và được ghi vào User.md.
- Correction có confidence đủ cao vẫn ghi đè ngay.

**Tác động.**

- Trên hai bộ dữ liệu, recall và token **không đổi**: mọi fact thật đều ≥ 0,7 (kiểm tra bằng cách in toàn bộ confidence).
- Có đúng một ứng viên dưới ngưỡng: "mình đang ở Huế để dùng ví dụ địa phương **nếu cần**" (0,5). Vì trùng fact đang có, nó chỉ làm mới `last_seen`.
- Lợi ích nằm ở các trường hợp ngoài bộ dữ liệu: fact sai không lọt vào `User.md`, nên không bị lặp lại ở mọi phiên và không tốn token ở mọi lượt.

**Rủi ro thêm vào.**

- **False negative:** nói một lần mà có rào đón ("chắc là mình sẽ ở Huế lâu dài") thì không được nhớ.
- **Hệ số không tổng quát:** các hệ số được chỉnh tay theo bộ dữ liệu. Danh sách từ rào đón dạng chuỗi con có thể bắt nhầm (ví dụ "nếu cần" ở cuối câu).
- **Tool live đi vòng qua ngưỡng:** ở chế độ live, tool `update_user_fact` của model ghi thẳng mà không qua ngưỡng này. Đây là điểm cần siết thêm nếu đưa lên production.

### 8.2. Memory decay (`fact_score`, `apply_decay`)

**Vấn đề giải quyết.** Fact đúng hôm nay có thể sai sau vài tháng mà người dùng không bao giờ đính chính. Thêm vào đó, mọi fact cũ vẫn bị gửi vào prompt mãi mãi.

**Cách làm.**

- Điểm hiện tại = `confidence × 0,5^(tuổi / half-life)`, trong đó tuổi tính bằng số lượt người dùng kể từ lần nhắc cuối.
- Half-life mặc định là 200 lượt (`MEMORY_HALF_LIFE_TURNS`), khác nhau theo loại fact:
  - `name` không decay;
  - `location` decay nhanh gấp đôi (×0,5);
  - `profession` ×0,75.
- Mỗi lần nhắc lại sẽ làm mới `last_seen` và tăng confidence (noisy-OR), tức tần suất nhắc cũng làm fact bền hơn.
- Theo điểm hiện tại:
  - dưới 0,35: chuyển sang mục `## Cần xác nhận lại (thông tin cũ)`. Agent vẫn trả lời nhưng ghi chú "(thông tin cũ, cần xác nhận lại)";
  - dưới 0,1: xoá khỏi `User.md`.

**Tác động.**

- Trên bộ standard (115 lượt), không fact nào bị chuyển sang stale. Fact cũ nhất (`pet`, nhắc lần cuối ở lượt 37) vẫn còn điểm 0,65, nên recall vẫn 100%.
- Thử với half-life 10 lượt: sau 10 lượt không nhắc, `location` và `profession` thành stale; sau 20–30 lượt thì bị xoá; `name` giữ nguyên. Nhắc lại "Mình vẫn ở Huế" thì `location` được ghi lại vào User.md.
- Lợi ích về token: `User.md` không tăng mãi theo thời gian. Fact không ai nhắc tới cuối cùng sẽ bị xoá, nên chi phí cố định mỗi lượt (mục 4) có giới hạn trên.

**Rủi ro thêm vào.**

- **Quên mất fact vẫn còn đúng:** fact ổn định nhưng hiếm khi được nhắc (dị ứng, tên con…) vẫn có thể bị xoá dù còn đúng. Cần danh sách "không decay" cho các trường quan trọng.
- **Thời gian tính theo lượt, không theo ngày:** người dùng chat nhiều sẽ thấy fact cũ đi nhanh hơn người chat ít.
- **Thêm file metadata:** có thêm một file cần đồng bộ với `User.md`. Nếu ai đó sửa tay `User.md`, fact không có metadata sẽ không bị decay. Code xử lý an toàn bằng cách giữ nguyên fact đó, nhưng hành vi giữa các fact không còn đồng nhất.

### 8.3. Hai guardrail có sẵn từ đầu

- **Conflict handling:** `upsert_fact` ghi đè chứ không ghi thêm, nên không bao giờ giữ đồng thời fact cũ và fact mới. Được kiểm chứng bằng `test_correction_keeps_latest_fact_only`.
- **Không lưu câu hỏi thành fact:** `is_question` chặn câu có "?", các câu mở đầu bằng yêu cầu recall, và cụm "nhắc lại giúp" ở giữa câu. Ngoài ra extractor bỏ qua mention bị phủ định và câu đùa. Được kiểm chứng bằng `test_questions_and_noise_are_not_stored`.

Test riêng cho bonus: `test_confidence_threshold_blocks_uncertain_facts`, `test_memory_decay_marks_then_prunes_old_facts`, `test_advanced_flags_stale_fact_in_answer`.

## 9. Câu chuyện tổng kết

1. Baseline không nhớ dài hạn: recall ở thread mới là 0%.
2. Advanced thêm `User.md` nên recall lên 100%, kể cả sau correction và nhiễu.
3. Hội thoại dài làm prompt của Baseline tăng theo bình phương số lượt (2.612 tokens/lượt ở cuối bộ stress).
4. Compact memory giữ prompt của Advanced trong dải khoảng 550–900 tokens/lượt, nên tổng prompt giảm 52,6%.
5. Đổi lại, hệ thống có thêm chi phí cố định ở hội thoại ngắn (+56,7%), rủi ro lưu sai fact vĩnh viễn, và mất chi tiết khi tóm tắt. Vì vậy cần guardrail: không lưu câu hỏi, câu đùa hay mention bị phủ định; correction ghi đè; confidence threshold trước khi ghi; decay để fact cũ được xác nhận lại hoặc bị xoá; danh sách và summary có giới hạn.
