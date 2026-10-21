# Lichess Puzzle Difficulty

Bài tập lớn học máy cho nhóm 3 người: dự đoán Rating bài tập cờ vua bằng Ridge,
Decision Tree và MLP. Đây là khung thực nghiệm ban đầu, chưa phải bài nộp hoàn chỉnh.

## Cấu trúc

```text
src/data.py           Đọc từng chunk và lấy mẫu toàn tệp bằng ưu tiên ngẫu nhiên
src/features.py       Trích xuất 23 đặc trưng từ FEN
src/models/linear.py  Thành viên 1: Ridge
src/models/tree.py    Thành viên 2: cây hồi quy
src/models/mlp.py     Thành viên 3: MLP
src/evaluation.py     MAE, RMSE, R²
src/run.py            Thực nghiệm chung
docs/PHAN_CONG.md     Phân công và checklist yêu cầu còn lại
notebook.ipynb        Notebook hiện có, giữ nguyên để nhóm phát triển
```

## Cài đặt và chạy

Python 3.10 trở lên. Chạy từ thư mục gốc:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.run --sample 20000
```

Dữ liệu lấy từ https://database.lichess.org/ (mục Puzzles).
Tải CSV Zstandard và giải nén thành `lichess_db_puzzle.csv` ở thư mục gốc,
hoặc truyền `--data DUONG_DAN`. Không đưa dữ liệu vào Git.
Bản dữ liệu cục bộ đã kiểm tra có 6.157.341 dòng, CSV 1.139.783.132 byte;
bản tải mới có thể khác. Ngày cập nhật bản nguồn cần được nhóm xác minh và ghi lại.

```powershell
zstd -d -k lichess_db_puzzle.csv.zst
python -m src.run --sample 1000 --models ridge tree
```

Kết quả nằm trong `outputs/metrics.csv`, gồm lỗi train và validation.
Mặc định lấy mẫu 20.000 bản ghi để thử nghiệm, không đọc toàn bộ CSV vào RAM.
Seed 42; các mô hình dùng cùng phép chia theo GameUrl. Tỷ lệ chia tính theo nhóm
ván nên tỷ lệ số bản ghi thực tế có thể chênh nhẹ. Mỗi lần chạy quét toàn bộ CSV.
Chuẩn hóa, mã hóa và giảm chiều chỉ fit trên train; chuẩn hóa Rating cũng chỉ fit
trên train để MLP ổn định, dự đoán và các độ đo vẫn ở đơn vị Rating gốc.

Runner hiện là baseline với siêu tham số cố định. Cần thêm tuning trên train,
phân tích phần dư, phân cụm, phân loại bổ sung, trực quan hóa, báo cáo và slide
theo `docs/PHAN_CONG.md`. Validation dùng để so sánh ban đầu; nếu tối ưu mô hình
nhiều lần cần tách thêm tập test cuối cùng. Cần kiểm tra trùng thế cờ giữa các ván.

Đầu vào chỉ dùng FEN. Không dùng RatingDeviation, Popularity, NbPlays, chuỗi lời giải
hoặc nhãn chiến thuật để dự đoán độ khó của thế cờ mới.
