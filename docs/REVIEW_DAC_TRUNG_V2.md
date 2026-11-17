# Kiểm tra bộ đặc trưng v2

Mẫu sạch 20000 dòng; seed 42; train 16000, validation 4000; 33 đặc trưng. Không có cột hằng.
Cùng mẫu và phép chia theo game_id; không dùng chung ván hoặc thế cờ.

| Mô hình | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 451.077 | 540.829 | -0.0000 |
| Ridge v2 | 440.282 | 528.582 | 0.0448 |

Chỉ là kiểm tra ban đầu ở tỷ lệ 80:20, chưa tuning và chưa có test cuối cùng.
Không so trực tiếp với số liệu nhánh khác nếu chưa xác nhận cùng mẫu và split.

Tái lập: `python -m src.prepare --sample 20000 --seed 42 --output outputs/features_v2`
rồi `python -m src.check_baseline --seed 42`.

File này là bản chụp kết quả của `src.check_baseline` ngày 17/11/2026; chạy lại lệnh
trên sẽ sinh `outputs/features_v2/baseline_review.md` với số liệu đo lại.
