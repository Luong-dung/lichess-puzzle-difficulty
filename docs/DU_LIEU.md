# Dữ liệu và tiền xử lý — phiên bản 2

## Kết quả lần chạy bộ đặc trưng v2 (17/11/2026)

Quét 6.157.341 dòng nguồn. Không thiếu ở chín cột từ PuzzleId đến GameUrl;
OpeningTags thiếu 4.932.426 dòng, DailyDate thiếu 6.155.554 dòng. Hai trường
này không dùng làm đầu vào nên không loại dòng chỉ vì thiếu chúng.

Mẫu ngẫu nhiên 20.000 dòng, seed 42: không có dòng thiếu trường bắt buộc,
Rating/URL không hợp lệ, FEN hoặc nước thiết lập lỗi, PuzzleId trùng hay
PuzzleFEN chuẩn hóa trùng; sau làm sạch còn 20.000 dòng. Đây là kết quả của mẫu,
không chứng minh toàn nguồn không có lỗi hoặc trùng.

Rating mẫu: trung bình 1.465,09; độ lệch chuẩn 546,52; min 399; Q1 1.037;
trung vị 1.411; Q3 1.865; max 3.065. Có 34 đặc trưng, không có cột hằng.
Train/validation ở ba tỷ lệ lần lượt 16.000/4.000, 14.000/6.000, 12.000/8.000
dòng; cả ba phép chia có giao ID ván và khóa thế cờ bằng 0.

## Chạy lại

```powershell
python -m src.prepare --sample 20000 --seed 42 --output outputs/features_v2
python -m unittest discover -s tests -v
python -m src.run --sample 20000 --seed 42
```

Cài requirements.txt và đặt CSV gốc ở thư mục gốc trước khi chạy. Kết quả trong outputs/ không commit. Không tái sử dụng features.csv cũ với 23 đặc trưng.

## Thế cờ người giải

Theo [tài liệu Lichess](https://database.lichess.org/#puzzles), FEN gốc là thế cờ trước nước đối thủ. Chỉ áp dụng nước đầu trong Moves để tạo PuzzleFEN; nước thứ hai mới bắt đầu lời giải. Không đọc các nước tiếp theo hoặc dùng số nước lời giải làm đặc trưng.

FEN phải có sáu trường, hợp lệ tĩnh theo python-chess. Nước đầu phải là UCI hợp lệ và nằm trong legal_moves. Kiểm tra này không chứng minh toàn bộ lịch sử ván hợp lệ. PuzzleFEN, khóa thế cờ và en_passant dùng thống nhất ô bắt tốt qua đường có hiệu lực pháp lý.

## Lấy mẫu và làm sạch

Quét toàn nguồn theo chunk 100.000 dòng; gán ưu tiên ngẫu nhiên seed 42, giữ 20.000 ưu tiên nhỏ nhất. Cùng tệp/thứ tự dòng/seed cho cùng mẫu. Làm sạch sau lấy mẫu, không bù dòng bị loại. Bộ đếm loại dòng tính tuần tự:

1. Thiếu PuzzleId/FEN/Moves/Rating/GameUrl.
2. Rating không hữu hạn hoặc không dương.
3. URL không phải HTTPS lichess.org với ID ván tám ký tự.
4. Trùng PuzzleId, giữ dòng đầu theo thứ tự ưu tiên lấy mẫu.
5. FEN hoặc nước thiết lập không hợp lệ.
6. Trùng khóa PuzzleFEN chuẩn hóa: bốn trường đầu, bỏ bộ đếm nước.

Thống kê toàn nguồn: số dòng, missing theo cột, dtype quan sát theo chunk. Xác thực chi tiết, lỗi/trùng và thống kê Rating/đặc trưng chỉ trên mẫu; không suy rộng thành toàn nguồn không có lỗi.
Đã xác nhận với giảng viên yêu cầu ít nhất 15 thuộc tính được tính sau trích xuất; bộ hiện tại
đạt (34 thuộc tính, đã bỏ hai cột số vua hằng).

## 34 đặc trưng đầu vào

| Tên | Số cột | Ý nghĩa |
|---|---:|---|
| white_p/n/b/r/q, black_p/n/b/r/q | 10 | Số tốt, mã, tượng, xe, hậu mỗi bên |
| white_material, black_material | 2 | Vật chất trọng số 1/3/3/5/9 |
| castle_K/Q/k/q | 4 | Quyền nhập thành |
| turn | 1 | Phân loại w/b: bên người giải tới lượt |
| en_passant | 1 | Có nước bắt tốt qua đường hợp lệ |
| halfmove, fullmove | 2 | Hai bộ đếm nước |
| material_difference | 1 | Vật chất trắng trừ đen |
| material_balance_to_move | 1 | Chênh lệch vật chất theo bên tới lượt: material_difference, đổi dấu khi đen tới lượt |
| white_pinned, black_pinned | 2 | Quân không phải vua bị ghim tuyệt đối vào vua |
| white_hanging, black_hanging | 2 | Quân không phải vua bị tấn công mà không được bảo vệ |
| white_attacked_squares, black_attacked_squares | 2 | Số ô bị mỗi bên tấn công, xấp xỉ độ hoạt động |
| white_king_zone_attacked, black_king_zone_attacked | 2 | Số ô kề vua bị đối phương tấn công |
| legal_moves | 1 | Số nước hợp lệ bên tới lượt |
| checking_moves | 1 | Số nước hợp lệ gây chiếu bên tới lượt |
| in_check | 1 | Bên tới lượt đang bị chiếu |
| phase_material | 1 | Trọng số mã/tượng/xe/hậu 1/1/2/4 hai bên, chặn 24 rồi chia 24 |

Đã bỏ hai cột số vua hằng. Mọi cột trừ turn là số. attacked_squares/hanging dùng quan hệ tấn công hình học của python-chess: quân bị ghim vẫn có thể tính là tấn công, không phải phân tích trao đổi tối ưu. Chỉ tính nước hợp lệ cho bên tới lượt; không đổi bên giả lập vì có thể tạo thế cờ không hợp lệ. phase_material là chỉ báo vật chất liên tục, không phải nhãn giai đoạn chính xác.

Không dùng Rating, RatingDeviation, Popularity, NbPlays, Themes hoặc phần lời giải sau nước thiết lập làm đầu vào. Bộ đặc trưng vẫn là baseline, không đảm bảo dự đoán tốt chỉ vì tăng số chiều.

## Chia tập và artifacts

Loại thế cờ trùng trước, rồi GroupShuffleSplit theo ID ván với validation 20%, 30%, 40%, seed 42. Kiểm tra giao ID ván và khóa PuzzleFEN bằng 0. Tỷ lệ dòng có thể lệch tỷ lệ nhóm. Cả ba mô hình dùng chung hàm chia. Mã hóa, chuẩn hóa, giảm chiều và biến đổi Rating chỉ fit trên train. Thế cờ gần giống chưa được coi là trùng; cần test độc lập nếu tuning nhiều lần.

Trong outputs/features_v2/: quality_report.json, sample.csv (gồm FEN gốc và PuzzleFEN), features.csv (cùng thứ tự hàng), feature_statistics.csv, split_20/30/40.csv, rating_distribution.png. Mã và tài liệu được commit, artifacts cục bộ không commit.
