# Kiểm tra bộ đặc trưng v2

Mẫu sạch 20000 dòng; seed 42; train 16000, validation 4000; 33 đặc trưng.
Cùng mẫu và phép chia theo game_id; không dùng chung ván hoặc thế cờ.

| Mô hình | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 451.077 | 540.829 | -0.0000 |
| Ridge v2 | 440.282 | 528.582 | 0.0448 |

Chỉ là kiểm tra ban đầu ở tỷ lệ 80:20, chưa tuning và chưa có test cuối cùng.
Không so trực tiếp với số liệu nhánh khác nếu chưa xác nhận cùng mẫu và split.

Tái lập: `python -m src.prepare --sample 20000 --seed 42 --output outputs/features_v2`
rồi `python -m src.check_baseline`.

Đã chạy 8 test đạt và smoke test 27 tổ hợp trên 300 dòng.
Mẫu 20.000 dòng: không loại lỗi/trùng, không có cột hằng, giao ván/thế cờ bằng 0 ở cả ba tỷ lệ.
