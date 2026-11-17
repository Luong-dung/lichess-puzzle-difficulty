# Thành viên 2: Random Forest và K-Means

Lường Tiến Dũng. Kết quả dưới đây đo trên mẫu 20.000 bài (seed 42) với bộ đặc trưng
hiện có trên `main` (23 đặc trưng đếm quân, nhập thành, lượt đi, số nước). **Cần chạy lại
sau khi khóa bộ đặc trưng mới (18/11)**; các con số sẽ thay đổi.

## Tái lập

```powershell
python -m src.prepare                 # tạo outputs/preprocessing/sample.csv (dùng làm cache)
python -m src.tune_random_forest      # ~3 phút, ghi outputs/random_forest/
python -m src.kmeans_analysis         # ~15 giây, ghi outputs/clustering/
```

Không có cache thì hai script tự quét CSV gốc. Tham số: `--n-iter`, `--folds`, `--k`.

## Random Forest

### Cách làm

- Chia train/validation 80/20 theo ván gốc (`split_groups`), giống các thành viên khác.
- Tìm siêu tham số bằng 20 cấu hình ngẫu nhiên trong lưới `n_estimators` {100, 200, 400},
  `max_depth` {8, 12, 16, 24, None}, `min_samples_leaf` {1, 2, 5, 10, 20},
  `max_features` {0.3, 0.5, 0.8, 1.0, sqrt}.
- Mỗi cấu hình chấm bằng GroupKFold 5 fold **chỉ trên train**, nhóm theo ván để không rò rỉ.
  Validation chỉ dùng một lần cho mô hình cuối.
- RF không cần chuẩn hóa: đặc trưng số giữ nguyên, `turn` mã hóa one-hot.

### Kết quả

Tham số tốt nhất: `n_estimators=400, max_depth=12, min_samples_leaf=10, max_features=0.5`
(CV RMSE 524,0 ± 6,0).

| Mô hình | Tập | MAE | RMSE | R² |
|---|---|---|---|---|
| Đoán trung bình train | validation | 451,1 | 540,8 | 0,000 |
| RF tốt nhất | train | 403,1 | 487,3 | 0,209 |
| RF tốt nhất | validation | 428,8 | 516,9 | 0,086 |

RF chỉ giảm RMSE khoảng 4% so với đoán trung bình: đặc trưng đếm quân mang rất ít
thông tin về độ khó chiến thuật của bài.

### Overfit (`hyperparameter_effect.png`)

RMSE trung bình theo `max_depth` trong 20 cấu hình:

| max_depth | train RMSE | CV RMSE |
|---|---|---|
| 8 | 506,8 | 526,1 |
| 12 | 474,5 | 524,9 |
| 16 | 365,1 | 530,2 |
| 24 | 430,7 | 528,5 |
| None | 312,5 | 539,6 |

Cây càng sâu, lá càng nhỏ thì train RMSE giảm mạnh nhưng CV RMSE tăng: cấu hình
`max_depth=None, min_samples_leaf=1` có train 218 nhưng CV 549, overfit rõ.
Giới hạn độ sâu 8–12 và lá ≥ 10 cho kết quả tốt và ổn định nhất.

### Learning curve (`learning_curve.png`)

Từ 1.280 lên 12.800 mẫu, CV RMSE giảm chậm từ 532,6 xuống 524,0, còn train RMSE tăng
từ 471 lên 485; khoảng cách hai đường thu hẹp. Thêm dữ liệu chỉ giúp ít; muốn cải thiện
nhiều cần đặc trưng tốt hơn.

### Phần dư (`residuals.png`, `residuals.csv`)

| Nhóm Rating thật (validation) | 399–959 | 959–1264 | 1264–1599 | 1599–1963 | 1963–3065 |
|---|---|---|---|---|---|
| Dự đoán trung bình | 1.398 | 1.447 | 1.479 | 1.494 | 1.528 |
| Phần dư trung bình | −645 | −334 | −53 | +278 | +731 |

Dự đoán bị co về quanh trung bình: bài dễ bị đoán khó hơn thực tế, bài khó bị đoán dễ
hơn. Đây là dấu hiệu mô hình thiếu thông tin, không phải lỗi cài đặt.

### Đặc trưng quan trọng (`feature_importance.png`)

| Hạng | Impurity (train) | Permutation (validation, RMSE tăng) |
|---|---|---|
| 1 | fullmove 0,189 | fullmove 15,4 |
| 2 | black_material 0,145 | black_material 15,1 |
| 3 | white_material 0,124 | white_material 9,8 |
| 4 | material_difference 0,114 | material_difference 8,5 |
| 5 | halfmove 0,075 | turn 7,4 |

Hai cách đo thống nhất: giai đoạn ván cờ (số nước, tổng vật chất) quan trọng nhất.
`halfmove` đứng thứ 5 theo impurity nhưng permutation ≈ 0 (−0,05 ± 0,37): cây chia
nhiều lần trên biến này vì nó có nhiều giá trị, nhưng nó không thực sự giúp dự đoán.

## K-Means

### Cách làm

Đặc trưng số và one-hot `turn` (bỏ cột hằng `white_k`, `black_k`), chuẩn hóa fit trên
train; K-Means `n_init=10` cho k = 2…10. Silhouette tính trên 5.000 mẫu train.
Validation được gán cụm bằng `predict` để kiểm tra Rating.

### Chọn k (`kmeans_elbow_silhouette.png`)

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Silhouette | 0,225 | **0,251** | 0,198 | 0,179 | 0,175 | 0,186 | 0,174 | 0,167 | 0,169 |

Elbow không có điểm gãy rõ; silhouette cao nhất ở k = 3, nhưng 0,25 vẫn là cấu trúc cụm yếu.

### Ý nghĩa các cụm (`kmeans_k3_centers.csv`)

| Cụm | Đặc điểm (tâm cụm) | Train | Rating TB (validation) |
|---|---|---|---|
| 0 | Tàn cuộc: vật chất mỗi bên ~11, fullmove ~42 | 5.357 | 1.455 |
| 1 | Trung cuộc: vật chất ~27, fullmove ~25 | 9.306 | 1.480 |
| 2 | Khai cuộc: vật chất ~34, fullmove ~13, còn quyền nhập thành | 1.337 | 1.391 |

K-Means tách thế cờ theo **giai đoạn ván**. Rating trung bình giữa các cụm chỉ chênh
khoảng 90 điểm trong khi độ lệch chuẩn trong cụm ~530; eta² trên validation = 0,002,
tức cụm giải thích 0,2% phương sai Rating. Hình chiếu PC1–PC2 (`kmeans_k3_pca.png`)
cũng không thấy vùng Rating tách biệt.

## Nhận xét chung

- Cả RF và K-Means cho thấy bộ đặc trưng hiện tại chủ yếu mô tả giai đoạn ván, gần như
  không phản ánh độ khó. RF tốt nhất chỉ đạt R² 0,086.
- Hướng cải thiện: đặc trưng chiến thuật từ python-chess (quân bị tấn công, chiếu,
  nước bắt quân hợp lệ...) dự kiến merge ngày 18/11, sau đó chạy lại hai script.
- So sánh SVD với PCA: dùng kết quả `pca_vs_svd.csv` của Doanh (PR #3) sau khi merge.
