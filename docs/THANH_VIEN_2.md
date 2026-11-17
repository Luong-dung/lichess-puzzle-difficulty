# Thành viên 2: Random Forest và K-Means

Lường Tiến Dũng. Kết quả đo trên mẫu 20.000 bài (seed 42) với bộ đặc trưng v2 đã khóa
ngày 18/11: 33 đặc trưng tính từ thế cờ người giải nhìn thấy (PuzzleFEN), gồm đếm quân,
vật chất và đặc trưng chiến thuật từ python-chess (`docs/DU_LIEU.md`).

## Tái lập

```powershell
python -m src.prepare --sample 20000 --seed 42 --output outputs/features_v2   # tạo cache
python -m src.tune_random_forest      # ~3 phút, ghi outputs/random_forest_v3/
python -m src.kmeans_analysis         # ~15 giây, ghi outputs/clustering_v2/
python -m src.rf_reduction            # ~3 phút, RF trên gốc / PCA / SVD, ghi outputs/rf_reduction/
```

Không có cache thì hai script tự quét CSV gốc. Tham số: `--n-iter`, `--folds`, `--k`, `--output`.

## Random Forest

### Cách làm

- Chia train/validation 80/20 theo ván gốc (`split_groups`), giống các thành viên khác.
- Thử 20 cấu hình ngẫu nhiên trong lưới `n_estimators` {100, 200, 400},
  `max_depth` {8, 12, 16, 24, None}, `min_samples_leaf` {1, 2, 5, 10, 20},
  `max_features` {0.3, 0.5, 0.8, 1.0, sqrt}.
- Mỗi cấu hình chấm bằng GroupKFold 5 fold **chỉ trên train**, nhóm theo ván để không rò rỉ.
  Validation chỉ dùng một lần cho mô hình cuối.
- **Chọn cấu hình bằng quy tắc 1-SE:** các cấu hình có CV RMSE không quá CV RMSE thấp nhất
  cộng một sai số chuẩn (độ lệch chuẩn giữa các fold / √5) được coi là ngang nhau; trong đó
  chọn cấu hình ít overfit nhất (khoảng cách CV − train nhỏ nhất).
- RF không cần chuẩn hóa: đặc trưng số giữ nguyên, `turn` mã hóa one-hot.

### Kết quả

Ngưỡng 1-SE là 509,0 (CV RMSE thấp nhất 505,6 ± 7,6 giữa các fold); 11/20 cấu hình nằm
trong ngưỡng. Cấu hình chọn: `n_estimators=400, max_depth=12, min_samples_leaf=20,
max_features=0.8` (CV RMSE 506,4).

| Mô hình | Tập | MAE | RMSE | R² |
|---|---|---|---|---|
| Đoán trung bình train | validation | 451,1 | 540,8 | 0,000 |
| RF (1-SE) | train | 384,7 | 467,5 | 0,272 |
| RF (1-SE) | validation | 410,7 | 498,4 | **0,151** |

Đối chiếu: nếu chọn cấu hình có CV thấp nhất (`max_depth=24, min_samples_leaf=5,
max_features=0.3`, 200 cây) thì validation R² = 0,156 nhưng train R² = 0,530, overfit nặng
hơn nhiều mà chỉ hơn 0,005 R². Với bộ đặc trưng cũ (23 đặc trưng tính sai trên thế cờ
trước nước thiết lập), RF tốt nhất chỉ đạt validation R² = 0,086.

### Overfit (`hyperparameter_effect.png`)

RMSE trung bình theo `max_depth` trong 20 cấu hình:

| max_depth | train RMSE | CV RMSE |
|---|---|---|
| 8 | 485,1 | 509,6 |
| 12 | 437,9 | 506,5 |
| 16 | 304,3 | 506,6 |
| 24 | 375,9 | 507,4 |
| None | 267,1 | 508,2 |

Cây sâu hoặc lá nhỏ làm train RMSE giảm mạnh mà CV gần như không đổi: cấu hình
`max_depth=None, min_samples_leaf=1` có train 188,6 nhưng CV 509,1. CV của mọi cấu hình
chỉ nằm trong khoảng 505–510, nên siêu tham số ảnh hưởng ít; giới hạn bởi đặc trưng.

### Learning curve (`learning_curve.png`)

Từ 1.280 lên 12.800 mẫu, CV RMSE giảm chậm từ 517,4 xuống 506,4, train RMSE gần như
đứng yên ~465. Thêm dữ liệu chỉ giúp ít; muốn cải thiện cần đặc trưng tốt hơn.

### Phần dư (`residuals.png`, `residuals.csv`)

| Nhóm Rating thật (validation) | 399–959 | 959–1264 | 1264–1599 | 1599–1963 | 1963–3065 |
|---|---|---|---|---|---|
| Dự đoán trung bình | 1.330 | 1.435 | 1.490 | 1.520 | 1.555 |
| Phần dư trung bình | −577 | −323 | −64 | +253 | +704 |

Dự đoán bị co về quanh trung bình: bài dễ bị đoán khó hơn thực tế, bài khó bị đoán dễ
hơn. Đây là dấu hiệu mô hình thiếu thông tin, không phải lỗi cài đặt.

### Đặc trưng quan trọng (`feature_importance.png`)

| Hạng | Impurity (train) | Permutation (validation, RMSE tăng) |
|---|---|---|
| 1 | checking_moves 0,186 | checking_moves 32,9 |
| 2 | fullmove 0,133 | fullmove 16,8 |
| 3 | phase_material 0,104 | phase_material 16,2 |
| 4 | legal_moves 0,077 | material_difference 6,1 |
| 5 | material_difference 0,067 | white_material 2,7 |

Hai cách đo thống nhất: số nước chiếu hợp lệ (đặc trưng chiến thuật mới) quan trọng nhất,
tiếp theo là giai đoạn ván (số nước, vật chất quân lớn).

## Random Forest trên dữ liệu giảm chiều: PCA và SVD

`python -m src.rf_reduction` (~3 phút, ghi `outputs/rf_reduction/`). Dùng đúng tham số RF
đã chọn ở trên. **PCA**: chuẩn hóa có trừ trung bình rồi PCA. **SVD**: chuẩn hóa không trừ
trung bình rồi TruncatedSVD, tức SVD chưa tâm hóa (nếu đã tâm hóa thì SVD trùng PCA).
Mọi bước fit trên train. Đầu vào 34 cột (33 đặc trưng, `turn` one-hot); k = 1/3 = 11.

### Giữ 1/3 số chiều (k = 11), ba tỷ lệ validation

| Validation | Gốc: RMSE / R² | PCA k=11: RMSE / R² | SVD k=11: RMSE / R² |
|---|---|---|---|
| 20% | 498,4 / 0,151 | 512,7 / 0,101 | 511,7 / 0,105 |
| 30% | 499,7 / 0,147 | 514,9 / 0,094 | 515,2 / 0,093 |
| 40% | 503,7 / 0,142 | 518,4 / 0,092 | 518,9 / 0,090 |

Giảm còn 11 chiều làm RF mất khoảng 1/3 R² ở cả ba tỷ lệ. PCA và SVD gần như ngang nhau
(chênh ≤ 1 điểm RMSE, đổi thứ hạng giữa các tỷ lệ).

### Quét số chiều k (validation 20%, `rf_pca_vs_svd.png`)

| k | 1 | 2 | 3 | 5 | 8 | 11 | 15 | 20 | 25 | 33 |
|---|---|---|---|---|---|---|---|---|---|---|
| PCA: phương sai giữ lại | 0,325 | 0,411 | 0,461 | 0,550 | 0,651 | 0,744 | 0,852 | 0,946 | 0,982 | 1,000 |
| SVD: phương sai giữ lại | 0,200 | 0,357 | 0,428 | 0,522 | 0,629 | 0,724 | 0,836 | 0,940 | 0,982 | 1,000 |
| PCA: RMSE validation | 537,0 | 531,7 | 526,9 | 517,0 | 513,6 | 512,7 | 512,1 | 507,9 | 507,0 | 507,4 |
| SVD: RMSE validation | 540,4 | 529,6 | 527,4 | 517,6 | 513,6 | 511,7 | 511,5 | 511,0 | 509,6 | 510,4 |

- Với cùng k, PCA luôn giữ nhiều phương sai hơn SVD, chênh nhiều nhất ở k nhỏ (k = 1:
  0,325 so với 0,200). SVD chưa tâm hóa dùng thành phần đầu để mô tả hướng trung bình
  của dữ liệu thay vì hướng biến thiên.
- Giữ nhiều phương sai hơn không kéo theo dự đoán tốt hơn rõ rệt: từ k = 8, hai cách
  chênh nhau dưới 4 điểm RMSE.
- **Ngay cả PCA giữ đủ 33 chiều (100% phương sai) vẫn kém dữ liệu gốc** (507,4 so với
  498,4). Phép xoay trộn các đặc trưng lại với nhau, trong khi cây quyết định chỉ cắt
  song song trục, nên mất các ngưỡng có nghĩa trên từng đặc trưng gốc (ví dụ
  `checking_moves`). Với RF, giảm chiều bằng PCA/SVD không có lợi; nên dùng đặc trưng gốc.

## K-Means

### Cách làm

Đặc trưng số và one-hot `turn` (bỏ cột hằng), chuẩn hóa fit trên train; K-Means
`n_init=10` cho k = 2…10. Silhouette tính trên 5.000 mẫu train. Validation được gán cụm
bằng `predict` để kiểm tra Rating.

### Chọn k (`kmeans_elbow_silhouette.png`)

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Silhouette | 0,218 | **0,221** | 0,159 | 0,135 | 0,135 | 0,124 | 0,119 | 0,120 | 0,103 |

Elbow không có điểm gãy rõ; silhouette cao nhất ở k = 3, nhưng 0,22 là cấu trúc cụm yếu.

### Ý nghĩa các cụm (`kmeans_k3_centers.csv`)

| Cụm | Đặc điểm (tâm cụm) | Train | Rating TB (validation) |
|---|---|---|---|
| 0 | Tàn cuộc: vật chất mỗi bên ~10, fullmove ~44, ~18 nước hợp lệ | 4.801 | 1.446 |
| 1 | Khai cuộc: vật chất ~34, fullmove ~13, còn quyền nhập thành | 1.259 | 1.403 |
| 2 | Trung cuộc: vật chất ~26, fullmove ~26, nhiều nước chiếu nhất (2,3) | 9.940 | 1.481 |

K-Means tách thế cờ theo **giai đoạn ván**. Rating trung bình giữa các cụm chỉ chênh
khoảng 80 điểm trong khi độ lệch chuẩn trong cụm ~530; eta² trên validation = 0,002,
tức cụm giải thích 0,2% phương sai Rating. Hình chiếu PC1–PC2 (`kmeans_k3_pca.png`)
cũng không thấy vùng Rating tách biệt.

## Nhận xét chung

- Sửa đặc trưng (đúng thế cờ + đặc trưng chiến thuật) gần như gấp đôi R² của RF
  (0,086 → 0,151); tuning siêu tham số chỉ thay đổi CV RMSE trong khoảng 5 điểm.
- RF vẫn chỉ giải thích ~15% phương sai Rating: độ khó nằm chủ yếu ở lời giải, mà
  lời giải không được dùng làm đặc trưng để tránh rò rỉ.
- Cụm K-Means mô tả giai đoạn ván, không phản ánh độ khó.
- PCA và SVD giữ 1/3 số chiều làm RF mất ~1/3 R²; hai cách gần ngang nhau về dự đoán.
- Còn lại: chạy lại cả ba script sau khi PR #6 (`material_balance_to_move`, 34 đặc trưng)
  được merge.
