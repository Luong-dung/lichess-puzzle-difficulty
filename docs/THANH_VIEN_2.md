# Thành viên 2: Random Forest, PCA/SVD và K-Means

Lường Tiến Dũng. Kết quả đo trên mẫu 20.000 bài (seed 42) với bộ đặc trưng đã khóa ngày
18/11: **34 đặc trưng** tính từ thế cờ người giải nhìn thấy (PuzzleFEN), gồm đếm quân,
vật chất, `material_balance_to_move` và đặc trưng chiến thuật từ python-chess
(`docs/DU_LIEU.md`).

## Tái lập

Chạy theo đúng thứ tự (bước 3 dùng tham số RF do bước 2 chọn):

```powershell
python -m src.prepare --sample 20000 --seed 42 --output outputs/features_v2   # 1. cache, ~1 phút
python -m src.tune_random_forest      # 2. ~3 phút, ghi outputs/random_forest_v4/
python -m src.rf_reduction            # 3. ~2 phút, RF trên gốc / PCA / SVD, ghi outputs/rf_reduction_v2/
python -m src.kmeans_analysis         # 4. ~15 giây, ghi outputs/clustering_v3/
```

Ba script chỉ đọc cache `outputs/features_v2/sample.csv` khi `quality_report.json` bên cạnh
có cùng `--seed` và `--sample`; không khớp thì in lý do và quét lại CSV gốc
(`src/sample_cache.py`). `rf_reduction` in rõ tham số RF lấy từ đâu và cảnh báo nếu
`best.json` được tune với seed hoặc số đặc trưng khác. Smoke test: `tests/test_member2.py`.

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

CV RMSE thấp nhất là 505,1 (`max_depth=12, min_samples_leaf=10, max_features=0.5`, 400 cây);
ngưỡng 1-SE là 508,6 và 17/20 cấu hình nằm trong ngưỡng. Cấu hình chọn:
`n_estimators=100, max_depth=8, min_samples_leaf=20, max_features=1.0`
(CV RMSE 508,4, khoảng cách CV − train chỉ 20 điểm).

| Mô hình | Tập | MAE | RMSE | R² |
|---|---|---|---|---|
| Đoán trung bình train | validation | 451,1 | 540,8 | 0,000 |
| RF (1-SE) | train | 404,2 | 490,5 | 0,199 |
| RF (1-SE) | validation | 413,0 | 500,2 | **0,145** |

So với các lần chạy trước trên cùng mẫu và phép chia:

| Bộ đặc trưng | Validation R² |
|---|---|
| 23 đặc trưng cũ (tính sai trên thế cờ trước nước thiết lập) | 0,086 |
| 33 đặc trưng (PR #2) | 0,151 |
| 34 đặc trưng (PR #6, bộ đã khóa) | 0,145 |

R² thấp hơn bộ 33 một chút không phải vì đặc trưng mới làm hại: 17/20 cấu hình nằm trong
ngưỡng 1-SE nên quy tắc chọn được cấu hình nông hơn hẳn (depth 8, train R² 0,199 so với
0,272 ở bộ 33). Đổi 0,006 R² validation lấy mô hình ít overfit hơn nhiều.

### Overfit (`hyperparameter_effect.png`)

RMSE trung bình theo `max_depth` trong 20 cấu hình. Mỗi giá trị là trung bình của các cấu
hình ngẫu nhiên khác nhau về tham số còn lại, nên đường train trong hình không đơn điệu;
dùng để thấy xu hướng, không phải ảnh hưởng riêng của từng tham số.

| max_depth | train RMSE | CV RMSE |
|---|---|---|
| 8 | 483,7 | 508,5 |
| 12 | 435,7 | 505,8 |
| 16 | 300,8 | 506,2 |
| 24 | 375,1 | 506,9 |
| None | 266,5 | 507,5 |

Cây sâu hoặc lá nhỏ làm train RMSE giảm mạnh mà CV gần như không đổi: cấu hình
`max_depth=None, min_samples_leaf=1` có train 188,4 nhưng CV 508,2. CV của mọi cấu hình
chỉ nằm trong khoảng 505–510, nên siêu tham số ảnh hưởng ít; giới hạn bởi đặc trưng.

### Learning curve (`learning_curve.png`)

Từ 1.280 lên 12.800 mẫu, CV RMSE giảm chậm từ 516,3 xuống 508,4, train RMSE tăng từ
470,9 lên 488,4; hai đường gần nhau. Thêm dữ liệu chỉ giúp ít; muốn cải thiện cần đặc
trưng tốt hơn.

### Phần dư (`residuals.png`, `residuals.csv`)

| Nhóm Rating thật (validation) | 399–959 | 959–1264 | 1264–1599 | 1599–1963 | 1963–3065 |
|---|---|---|---|---|---|
| Dự đoán trung bình | 1.337 | 1.437 | 1.487 | 1.513 | 1.551 |
| Phần dư trung bình | −583 | −325 | −62 | +260 | +708 |

Dự đoán bị co về quanh trung bình: bài dễ bị đoán khó hơn thực tế, bài khó bị đoán dễ
hơn. Đây là dấu hiệu mô hình thiếu thông tin, không phải lỗi cài đặt.

### Đặc trưng quan trọng (`feature_importance.png`)

| Hạng | Impurity (train) | Permutation (validation, RMSE tăng) |
|---|---|---|
| 1 | checking_moves 0,259 | checking_moves 33,8 |
| 2 | fullmove 0,138 | phase_material 17,9 |
| 3 | phase_material 0,129 | fullmove 15,0 |
| 4 | material_balance_to_move 0,094 | material_balance_to_move 9,3 |
| 5 | legal_moves 0,053 | white_material 2,3 |

Hai cách đo thống nhất: số nước chiếu hợp lệ quan trọng nhất, tiếp theo là giai đoạn ván
(vật chất quân lớn, số nước). Đặc trưng mới `material_balance_to_move` đứng thứ 4 ở cả hai
cách đo, thay chỗ `material_difference` (tính theo quân trắng).

## Random Forest trên dữ liệu giảm chiều: PCA và SVD

Dùng đúng tham số RF đã chọn ở trên. **PCA**: chuẩn hóa có trừ trung bình rồi PCA.
**SVD**: chuẩn hóa không trừ trung bình rồi TruncatedSVD, tức SVD chưa tâm hóa (nếu đã tâm
hóa thì SVD trùng PCA). Mọi bước fit trên train. Đầu vào 35 cột (34 đặc trưng, `turn`
one-hot); k = 1/3 = 11.

### Giữ 1/3 số chiều (k = 11), ba tỷ lệ validation

| Validation | Gốc: RMSE / R² | PCA k=11: RMSE / R² | SVD k=11: RMSE / R² |
|---|---|---|---|
| 20% | 500,2 / 0,145 | 516,0 / 0,090 | 513,9 / 0,097 |
| 30% | 501,8 / 0,140 | 516,0 / 0,090 | 516,3 / 0,089 |
| 40% | 505,5 / 0,136 | 519,0 / 0,089 | 519,8 / 0,087 |

Tham số RF được tune trên train của phép chia 20%. Ở phép chia 30% và 40%, lần lượt
khoảng 33% và 50% số dòng validation đã nằm trong tập train dùng để tune, nên hai dòng đó
hơi lạc quan; so sánh giữa gốc, PCA và SVD trong cùng một dòng vẫn công bằng.

Giảm còn 11 chiều làm RF mất khoảng 1/3 R² ở cả ba tỷ lệ. PCA và SVD gần như ngang nhau
(chênh ≤ 2,1 điểm RMSE, đổi thứ hạng giữa các tỷ lệ).

### Quét số chiều k (validation 20%, `rf_pca_vs_svd.png`)

| k | 1 | 2 | 3 | 5 | 8 | 11 | 15 | 20 | 25 | 34 |
|---|---|---|---|---|---|---|---|---|---|---|
| PCA: phương sai giữ lại | 0,315 | 0,400 | 0,450 | 0,538 | 0,640 | 0,730 | 0,835 | 0,937 | 0,978 | 1,000 |
| SVD: phương sai giữ lại | 0,194 | 0,348 | 0,416 | 0,509 | 0,618 | 0,710 | 0,820 | 0,928 | 0,977 | 1,000 |
| PCA: RMSE validation | 536,0 | 529,3 | 526,9 | 518,7 | 516,4 | 516,0 | 515,2 | 512,7 | 512,5 | 512,3 |
| SVD: RMSE validation | 537,4 | 529,5 | 526,9 | 519,2 | 515,4 | 513,9 | 514,1 | 514,6 | 513,5 | 513,4 |

- Với cùng k, PCA luôn giữ nhiều phương sai hơn SVD, chênh nhiều nhất ở k nhỏ (k = 1:
  0,315 so với 0,194). SVD chưa tâm hóa dùng thành phần đầu để mô tả hướng trung bình
  của dữ liệu thay vì hướng biến thiên.
- Giữ nhiều phương sai hơn không kéo theo dự đoán tốt hơn rõ rệt: từ k = 8, hai cách
  chênh nhau dưới 2,5 điểm RMSE.
- **Ngay cả PCA giữ đủ 34 chiều (100% phương sai) vẫn kém dữ liệu gốc** (512,3 so với
  500,2). Phép xoay trộn các đặc trưng lại với nhau, trong khi cây quyết định chỉ cắt
  song song trục, nên mất các ngưỡng có nghĩa trên từng đặc trưng gốc (ví dụ
  `checking_moves`). Với RF, giảm chiều bằng PCA/SVD không có lợi; nên dùng đặc trưng gốc.

## K-Means

### Cách làm

Đặc trưng số và one-hot `turn`, bỏ cột hằng (xác định trên train), chuẩn hóa fit trên
train; K-Means `n_init=10` cho k = 2…10. Silhouette tính trên 5.000 mẫu train.
Validation được gán cụm bằng `predict` để kiểm tra Rating.

### Chọn k (`kmeans_elbow_silhouette.png`)

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Silhouette | 0,214 | **0,217** | 0,156 | 0,135 | 0,138 | 0,118 | 0,106 | 0,112 | 0,109 |

Elbow không có điểm gãy rõ; silhouette cao nhất ở k = 3, nhưng 0,22 là cấu trúc cụm yếu.

### Ý nghĩa các cụm (`kmeans_k3_centers.csv`)

| Cụm | Đặc điểm (tâm cụm) | Train | Rating TB (validation) |
|---|---|---|---|
| 0 | Trung cuộc: vật chất trắng ~26, fullmove ~26, nhiều nước chiếu nhất (2,3) | 9.965 | 1.479 |
| 1 | Khai cuộc: vật chất ~34, fullmove ~13, còn quyền nhập thành | 1.253 | 1.400 |
| 2 | Tàn cuộc: vật chất ~10, fullmove ~44, ~18 nước hợp lệ | 4.782 | 1.450 |

K-Means tách thế cờ theo **giai đoạn ván**. Rating trung bình giữa các cụm chỉ chênh
khoảng 80 điểm trong khi độ lệch chuẩn trong cụm ~490–585; eta² trên validation = 0,002,
tức cụm giải thích 0,2% phương sai Rating. Hình chiếu PC1–PC2 (`kmeans_k3_pca.png`)
cũng không thấy vùng Rating tách biệt.

## Nhận xét chung

- Sửa đặc trưng (đúng thế cờ + đặc trưng chiến thuật) gần như gấp đôi R² của RF
  (0,086 → 0,145–0,151); tuning siêu tham số chỉ thay đổi CV RMSE trong khoảng 5 điểm.
- RF vẫn chỉ giải thích ~15% phương sai Rating: độ khó nằm chủ yếu ở lời giải, mà
  lời giải không được dùng làm đặc trưng để tránh rò rỉ.
- PCA và SVD giữ 1/3 số chiều làm RF mất ~1/3 R²; hai cách gần ngang nhau về dự đoán.
- Cụm K-Means mô tả giai đoạn ván, không phản ánh độ khó.
