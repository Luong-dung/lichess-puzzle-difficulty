# Thành viên 1: EDA, PCA và Ridge

## Chạy lại

```powershell
python -m src.eda --sample 20000 --seed 42              # -> outputs/eda/
python -m src.linear_analysis --sample 20000 --seed 42  # -> outputs/linear/
python -m unittest discover -s tests -v
```

Cả hai dùng cùng `load_sample`, `split_groups` và tiền xử lý với `src/run.py`
(`src/preprocessing.py`). PCA, chuẩn hóa và chọn alpha chỉ fit trên train.

## EDA (mẫu 20.000 dòng, seed 42, lần chạy 21/10/2026)

- Cột hằng: `white_k`, `black_k`. Bên tới lượt: đen 51,4%, trắng 48,6%.
- Tương quan Spearman đơn biến với Rating đều rất yếu, |ρ| ≤ 0,07. Mạnh nhất là
  `fullmove` (+0,062), sau đó là bốn quyền nhập thành (−0,03 đến −0,04).
- Biểu đồ: `rating_distribution.png`, `feature_histograms.png`,
  `correlation_heatmap.png`, `rating_correlation.png`, `rating_by_material_and_move.png`.

## PCA (fit trên train của phép chia 20%, 16.000 dòng, 24 chiều sau one-hot)

| | Giá trị |
|---|---|
| PC1–PC6 | 38,3%; 11,0%; 6,2%; 5,6%; 5,1%; 4,9% |
| k = 8 (1/3 số chiều, như runner) | 80,8% phương sai |
| Số thành phần cho 80% / 90% / 95% | 8 / 11 / 13 |
| TruncatedSVD với k = 8 | 80,8% |

PCA và TruncatedSVD trong runner gần như trùng nhau vì dữ liệu đã chuẩn hóa
(trung bình 0) trước khi giảm chiều. Chỉ hai cột one-hot `turn` không được tâm
hóa, nên SVD không khác PCA đáng kể.

Để so sánh có nội dung, `src.eda` chạy thêm SVD trên dữ liệu **chưa tâm hóa**
(chia độ lệch chuẩn, không trừ trung bình). Cả hai fit trên train, sai số tái tạo
đo trên validation (`pca_vs_svd_variance.png`, `pca_vs_svd.csv`):

| k | PCA: phương sai train | SVD chưa tâm hóa: phương sai train | PCA: phần không tái tạo (val) | SVD chưa tâm hóa: phần không tái tạo (val) |
|---:|---:|---:|---:|---:|
| 1 | 38,3% | 21,7% | 63,1% | 80,5% |
| 4 | 61,1% | 56,8% | 40,8% | 45,1% |
| 8 | 80,8% | 77,2% | 19,0% | 23,6% |

- Ở cùng k, PCA luôn tái tạo tốt hơn. SVD chưa tâm hóa phải dùng thành phần đầu
  để mô tả vector trung bình. Thành phần này gần như không chứa thông tin phân
  biệt các bài, nên SVD cần 9 chiều mới đạt sai số của PCA ở k = 8.
- Hai đường gặp nhau từ khoảng 15 chiều. PCA đạt 100% ở 18 chiều vì 24 cột đầu
  vào có 6 chiều dư thừa: 2 cột vua hằng, hai cột tổng vật chất và
  `material_difference` là tổ hợp tuyến tính của số quân, cùng hai cột one-hot
  `turn` có tổng bằng 1.
- Khi chiếu validation lên hai thành phần đầu (`pca_vs_svd_projection.png`), cả
  hai cách đều không tách được vùng Rating khác nhau. Kết luận về giới hạn của bộ
  đặc trưng không đổi.
- SVD chưa tâm hóa phù hợp hơn với dữ liệu thưa (ví dụ one-hot 64 ô × 12 loại
  quân), nơi việc trừ trung bình phá vỡ tính thưa. Hướng này để trao đổi với
  thành viên 2 (phụ trách SVD).

Khi chiếu tập validation lên PC1–PC4 (`pca_components_by_rating.png`), không
thấy vùng nào có Rating trung bình khác biệt rõ. Các thành phần chính mô tả
lượng vật chất và giai đoạn ván, không mô tả độ khó.

## Ridge và OLS

Alpha được chọn trên lưới 10⁻³…10⁴ bằng GroupKFold 5 fold theo ván, chỉ dùng
train. Đường CV gần như phẳng khi alpha ≤ 10³. Alpha tốt nhất là 10^2,5 (biểu
diễn gốc) và 10³ (PCA/SVD).

| Validation | Biểu diễn | Mô hình | RMSE train | RMSE val | MAE val | R² val |
|---|---|---|---:|---:|---:|---:|
| 20% | gốc | Trung bình train | 547,9 | 540,8 | 451,1 | 0,000 |
| 20% | gốc | OLS | 544,6 | 536,6 | 447,6 | 0,016 |
| 20% | gốc | Ridge | 544,6 | 536,7 | 447,7 | 0,015 |
| 20% | PCA k=8 | Ridge | 546,1 | 538,6 | 449,4 | 0,008 |
| 30% | gốc | Ridge | 545,6 | 537,4 | 447,7 | 0,014 |
| 40% | gốc | Ridge | 545,0 | 540,3 | 449,8 | 0,013 |

Chi tiết đủ 27 dòng trong `outputs/linear/linear_summary.csv`.

## Nhận xét

- **Không overfit, mà underfit nặng.** RMSE train và validation chênh dưới 10
  Rating, learning curve phẳng. Mô hình chỉ tốt hơn dự đoán hằng số khoảng 4
  Rating RMSE (R² ≈ 0,015).
- **Phần dư co về trung bình.** Dự đoán chỉ nằm trong khoảng 1.300–1.750.
  Bài dễ (< 800) bị dự đoán cao hơn thực tế khoảng 800, bài khó (> 2.200) bị dự
  đoán thấp hơn khoảng 950 (`ridge_residuals.png`).
- Nguyên nhân là 23 đặc trưng đếm quân và vật chất gần như không chứa thông tin
  về độ khó chiến thuật. Ridge và PCA không khắc phục được, vì giới hạn nằm ở
  đặc trưng chứ không ở mô hình. Đề xuất cho cả nhóm: thêm đặc trưng từ
  python-chess như số nước hợp lệ, bên tới lượt có bị chiếu không, số quân bị
  tấn công hoặc không được bảo vệ, độ an toàn của vua, chênh lệch vật chất theo
  góc nhìn bên tới lượt. Sau đó chạy lại cả ba mô hình. Những đặc trưng này vẫn
  chỉ lấy từ FEN nên không vi phạm quy tắc đầu vào.
