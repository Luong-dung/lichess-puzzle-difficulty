# Dữ liệu và tiền xử lý

## Kết quả lần chạy ngày 21/10/2026

Quét 6.157.341 dòng nguồn. Không thiếu ở chín cột từ PuzzleId đến GameUrl;
OpeningTags thiếu 4.932.426 dòng, DailyDate thiếu 6.155.554 dòng.
Hai trường này không dùng làm đầu vào nên không loại dòng chỉ vì thiếu chúng.

Mẫu ngẫu nhiên 20.000 dòng, seed 42: không phát hiện thiếu trường bắt buộc,
Rating/URL không hợp lệ, FEN lỗi, PuzzleId trùng hoặc thế cờ chuẩn hóa trùng.
Sau làm sạch còn 20.000 dòng. Đây là kết quả của mẫu, không chứng minh toàn
nguồn không có lỗi hoặc trùng.

Rating mẫu: trung bình 1.465,09; độ lệch chuẩn 546,52; min 399; Q1 1.037;
trung vị 1.411; Q3 1.865; max 3.065. Hai cột hằng là white_k và black_k.
Các tập train/validation lần lượt có 16.000/4.000, 14.000/6.000, 12.000/8.000
dòng. Cả ba phép chia có giao ID ván và khóa thế cờ bằng 0.

Đã đạt 5 kiểm tra tự động về FEN, lỗi/trùng, đặc trưng, lấy mẫu tái lập và chia
tập; chạy thử cả ba mô hình qua 27 tổ hợp trên mẫu nhỏ 300 dòng. Đây là kiểm tra
luồng chương trình, chưa phải kết quả đánh giá chất lượng mô hình cuối cùng.

## Chạy lại

Từ thư mục gốc, sau khi cài `requirements.txt` và đặt CSV tại thư mục gốc:

```powershell
python -m src.prepare --sample 20000 --seed 42
python -m unittest discover -s tests -v
python -m src.run --sample 20000 --seed 42 --models ridge random_forest mlp
```

`prepare` quét toàn bộ CSV theo chunk 100.000 dòng. Mỗi dòng được gán ưu tiên
ngẫu nhiên, giữ 20.000 ưu tiên nhỏ nhất trên toàn tệp. Cùng tệp, thứ tự dòng và
seed sẽ tạo cùng mẫu. Lấy mẫu trước, làm sạch sau nên số dòng còn lại có thể ít
hơn số yêu cầu; không bù thêm dòng sau khi loại lỗi/trùng.

## Phạm vi thống kê

- Toàn nguồn: số dòng, số giá trị thiếu theo từng cột, các dtype pandas quan sát
  được theo chunk. Dtype suy luận không thay thế kiểm tra ngữ nghĩa.
- Chỉ trên mẫu: FEN, Rating, URL, trùng PuzzleId/thế cờ, thống kê đặc trưng và Rating.
  Không suy rộng số lỗi hoặc số bản ghi trùng của mẫu thành số của toàn nguồn.
- Bộ đếm loại dòng tính tuần tự: một dòng chỉ được tính ở lý do bị loại đầu tiên.

## Quy tắc làm sạch

1. Loại dòng thiếu PuzzleId, FEN, Rating hoặc GameUrl.
2. Ép Rating sang số; loại giá trị không hữu hạn hoặc không dương.
3. Trích ID ván 8 ký tự từ URL HTTPS của lichess.org; loại URL không phù hợp.
4. Loại PuzzleId trùng, giữ dòng đầu theo thứ tự ưu tiên lấy mẫu.
5. FEN phải có sáu trường, đọc được bằng python-chess và `Board.is_valid()`.
   Kiểm tra này xác nhận tính hợp lệ tĩnh, không chứng minh thế cờ có thể đạt tới
   từ vị trí ban đầu bằng một lịch sử ván hợp lệ.
6. Chuẩn hóa thế cờ bằng `board.fen(en_passant="legal")`, giữ bốn trường đầu:
   bàn cờ, bên tới lượt, quyền nhập thành, ô bắt tốt qua đường có hiệu lực.
   Bỏ hai bộ đếm nước khi so trùng; giữ một dòng cho mỗi thế cờ chuẩn hóa.

## 23 đặc trưng đầu vào

| Tên | Số cột | Kiểu | Ý nghĩa |
|---|---:|---|---|
| `white_p/n/b/r/q/k` | 6 | Số nguyên | Số tốt/mã/tượng/xe/hậu/vua trắng |
| `black_p/n/b/r/q/k` | 6 | Số nguyên | Số quân tương ứng của đen |
| `white_material`, `black_material` | 2 | Số | Tổng giá trị với trọng số 1/3/3/5/9/0 |
| `castle_K/Q/k/q` | 4 | Nhị phân | Bốn quyền nhập thành |
| `turn` | 1 | Phân loại | Bên tới lượt: w hoặc b |
| `en_passant` | 1 | Nhị phân | FEN có khai báo ô bắt tốt qua đường |
| `halfmove`, `fullmove` | 2 | Số nguyên | Hai bộ đếm trong FEN |
| `material_difference` | 1 | Số | Vật chất trắng trừ đen |

Đây là bộ đặc trưng baseline, chưa mô tả đầy đủ chiến thuật hay cấu trúc vị trí.
Số vua có thể là cột hằng; báo cáo chất lượng liệt kê cột hằng để nhóm cân nhắc
loại hoặc thay bằng đặc trưng khác khi phân tích giảm chiều. Không dùng Rating,
RatingDeviation, Popularity, NbPlays, Moves hoặc Themes làm đầu vào. Cần xác nhận
với giảng viên yêu cầu 15 thuộc tính được tính sau trích xuất.

## Chia dữ liệu và chống rò rỉ

Loại thế cờ trùng trong mẫu trước, sau đó GroupShuffleSplit theo ID ván ở ba tỷ lệ
validation 20%, 30%, 40%, seed 42. Tỷ lệ theo nhóm nên số dòng có thể chênh nhẹ.
Hàm chia kiểm tra giao của ID ván và khóa thế cờ giữa hai tập đều rỗng, báo lỗi
nếu có rò rỉ. Cả ba mô hình sử dụng cùng hàm chia và cùng mẫu.

Mã hóa, chuẩn hóa, giảm chiều và chuẩn hóa nhãn chỉ fit trên train trong Pipeline.
Các thế cờ gần giống chưa được coi là trùng. Validation phục vụ so sánh ban đầu;
nếu tuning nhiều lần cần tập test cuối cùng độc lập.

## Kết quả sinh ra

Trong `outputs/preprocessing/` (được Git bỏ qua):

- `quality_report.json`: phạm vi kiểm tra, bộ đếm, thống kê Rating, cột hằng và số dòng các tập.
- `sample.csv`, `features.csv`: mẫu sạch và đặc trưng có cùng thứ tự hàng.
- `feature_statistics.csv`: thống kê các đặc trưng.
- `split_20.csv`, `split_30.csv`, `split_40.csv`: PuzzleId và nhãn train/validation để đối chiếu.
- `rating_distribution.png`: biểu đồ Rating của mẫu sạch, ghi cỡ mẫu và seed.

Không commit dữ liệu, kết quả hoặc biểu đồ sinh tự động. Phần phân cụm, phân loại
bổ sung, tuning và trực quan hóa PCA/SVD thuộc các công việc tiếp theo.
