# Thành viên 1: EDA, PCA, Ridge và Logistic Regression

## Chạy lại

```powershell
python -m src.eda --sample 20000 --seed 42                # -> outputs/eda/
python -m src.linear_analysis --sample 20000 --seed 42    # -> outputs/linear/
python -m src.logistic_analysis --sample 20000 --seed 42  # -> outputs/logistic/
python -m src.compare                                     # -> outputs/comparison/
python -m unittest discover -s tests -v
```

`src.compare` không huấn luyện lại mà chỉ gom file của cả ba thành viên, nên cần chạy
các script khác trước (thứ tự ghi ở đầu `src/compare.py`).

Cả hai dùng cùng `load_sample`, `split_groups` và tiền xử lý với `src/run.py`
(`src/preprocessing.py`). PCA, chuẩn hóa và chọn alpha chỉ fit trên train.

Kết quả dưới đây là lần chạy ngày 17/11/2026 trên bộ đặc trưng v2: thế cờ người
giải (PuzzleFEN) và 34 đặc trưng (`docs/DU_LIEU.md`). Số liệu bộ 23 đặc trưng cũ
(21/10) nằm trong lịch sử Git và không còn dùng để so sánh.

## EDA (mẫu 20.000 dòng, seed 42)

- Không có cột hằng. Bên tới lượt (người giải): trắng 51,4%, đen 48,6%.
- Tương quan Spearman đơn biến với Rating vẫn yếu, |ρ| ≤ 0,10. Mạnh nhất là
  `material_balance_to_move` (+0,093), `in_check` (+0,071), `halfmove` (+0,066),
  `white_q` (+0,065), hai cột `*_king_zone_attacked` (+0,06) và `fullmove` (+0,063).
  Đặc trưng mới theo góc nhìn bên tới lượt có tương quan cao nhất, trong khi
  `material_difference` theo góc nhìn trắng gần như không tương quan.
- Biểu đồ: `rating_distribution.png`, `feature_histograms.png`,
  `correlation_heatmap.png`, `rating_correlation.png`, `rating_by_material_and_move.png`.

## PCA (fit trên train của phép chia 20%, 16.000 dòng, 35 chiều sau one-hot)

| | Giá trị |
|---|---|
| PC1–PC6 | 31,5%; 8,5%; 5,1%; 4,6%; 4,2%; 3,6% |
| k = 11 (1/3 số chiều, như runner) | 73,0% phương sai |
| Số thành phần cho 80% / 90% / 95% | 14 / 18 / 22 |
| TruncatedSVD với k = 11 | 73,0% |

So với bộ cũ, phương sai trải đều hơn: PC1 giảm từ 38,3% xuống 31,5% và cần
14 thay vì 8 thành phần để đạt 80%. Các đặc trưng chiến thuật mang thông tin
không trùng với đếm quân.

PCA và TruncatedSVD trong runner gần như trùng nhau vì dữ liệu đã chuẩn hóa
(trung bình 0) trước khi giảm chiều; chỉ hai cột one-hot `turn` không được tâm hóa.

Để so sánh có nội dung, `src.eda` chạy thêm SVD trên dữ liệu **chưa tâm hóa**
(chia độ lệch chuẩn, không trừ trung bình). Cả hai fit trên train, sai số tái tạo
đo trên validation (`pca_vs_svd_variance.png`, `pca_vs_svd.csv`):

| k | PCA: phương sai train | SVD chưa tâm hóa: phương sai train | PCA: phần không tái tạo (val) | SVD chưa tâm hóa: phần không tái tạo (val) |
|---:|---:|---:|---:|---:|
| 1 | 31,5% | 19,4% | 69,4% | 81,7% |
| 4 | 49,6% | 46,5% | 51,8% | 55,0% |
| 11 | 73,0% | 71,0% | 26,9% | 29,1% |

- Ở cùng k, PCA luôn tái tạo tốt hơn. SVD chưa tâm hóa dùng thành phần đầu để
  mô tả vector trung bình, nên cần 12 chiều mới đạt sai số của PCA ở k = 11.
- PCA đạt 100% ở 30 chiều vì 35 cột đầu vào có khoảng 5 chiều dư thừa: hai cột
  tổng vật chất và `material_difference` là tổ hợp tuyến tính của số quân,
  `phase_material` gần như tuyến tính theo số quân (chỉ bị chặn ở 24), và hai cột
  one-hot `turn` có tổng bằng 1. `material_balance_to_move` không dư thừa vì phụ
  thuộc cả `turn`.
- Khi chiếu validation lên hai thành phần đầu (`pca_vs_svd_projection.png`) và
  PC1–PC4 (`pca_components_by_rating.png`), không thấy vùng Rating tách biệt rõ.
- SVD chưa tâm hóa phù hợp hơn với dữ liệu thưa (ví dụ one-hot 64 ô × 12 loại
  quân). Hướng này để trao đổi với thành viên 2 (phụ trách SVD).

## Ridge và OLS

Alpha được chọn trên lưới 10⁻³…10⁴ bằng GroupKFold 5 fold theo ván, chỉ dùng
train. Đường CV gần như phẳng khi alpha ≤ 10². Alpha tốt nhất là 10² (biểu diễn
gốc) và 10^2,5–10³ (PCA/SVD).

| Validation | Biểu diễn | Mô hình | RMSE train | RMSE val | MAE val | R² val |
|---|---|---|---:|---:|---:|---:|
| 20% | gốc | Trung bình train | 547,9 | 540,8 | 451,1 | 0,000 |
| 20% | gốc | OLS | 532,2 | 524,2 | 435,4 | 0,061 |
| 20% | gốc | Ridge | 532,2 | 524,2 | 435,5 | 0,061 |
| 20% | PCA k=11 | Ridge | 541,0 | 532,8 | 444,0 | 0,029 |
| 20% | SVD k=11 | Ridge | 541,0 | 532,8 | 444,0 | 0,030 |
| 30% | gốc | Ridge | 532,6 | 526,2 | 436,5 | 0,054 |
| 40% | gốc | Ridge | 531,7 | 529,5 | 438,5 | 0,052 |

Chi tiết đủ 27 dòng trong `outputs/linear/linear_summary.csv`.

So với bộ 23 đặc trưng cũ (Ridge 20%: RMSE val 536,7, R² 0,015), bộ v2 giảm
RMSE khoảng 12,5 Rating và tăng R² gấp 4 lần, nhưng vẫn thấp tuyệt đối.

Hệ số chuẩn hóa lớn nhất (Rating trên 1 độ lệch chuẩn, `ridge_coefficients.csv`):
`material_balance_to_move` +71,9; `black_attacked_squares` −69,0; `fullmove` +65,0;
`white_attacked_squares` −61,7; `in_check` +57,7; `white_q` +49,7; `legal_moves` +48,5.
Các hệ số bị ảnh hưởng bởi tương quan giữa đặc trưng nên chỉ đọc như xu hướng,
không phải tác động nhân quả.

## Nhận xét

- **Không overfit, mà underfit.** RMSE train và validation chênh dưới 10 Rating;
  learning curve gần phẳng từ 5.600 dòng train (`ridge_learning_curve.png`).
  Thêm dữ liệu không giúp nhiều, giới hạn nằm ở đặc trưng và tính tuyến tính.
- **Phần dư co về trung bình.** Nhóm Rating thấp nhất (399–790) bị dự đoán cao
  hơn thực tế trung bình 762, nhóm cao nhất (2.208–3.065) bị dự đoán thấp hơn 936
  (`ridge_residuals.png`, `ridge_residuals_by_rating.csv`).
- **PCA/SVD với 1/3 số chiều làm giảm R² một nửa** (0,061 → 0,029). Các thành
  phần giữ nhiều phương sai nhất mô tả lượng vật chất, không phải các tín hiệu
  yếu liên quan đến Rating, nên giảm chiều không giám sát bỏ mất thông tin hữu ích.
- Bộ đặc trưng v2 tốt hơn rõ nhưng vẫn chỉ mô tả bề mặt thế cờ. Độ khó puzzle
  phụ thuộc vào độ sâu chiến thuật (nước yên lặng, hy sinh, số nước lời giải)
  mà đặc trưng tĩnh từ FEN chỉ phản ánh gián tiếp. Mô hình phi tuyến (Random
  Forest, MLP) có thể khai thác tương tác giữa các đặc trưng tốt hơn Ridge.

## Logistic Regression (phân loại khoảng Rating)

Dùng chung nhãn với Naive Bayes của Thành viên 3: ngưỡng phân vị học trên train
(`fit_rating_bins`). Ở phép chia 20%, 3 khoảng cắt tại 1.154 và 1.699; 4 khoảng
cắt tại 1.036, 1.409 và 1.865. Softmax đa lớp, phạt L2. C được chọn trên lưới
10⁻⁴…10² bằng GroupKFold 5 fold theo ván trên train, tiêu chí macro-F1. Đối chứng
là bộ đoán lớp đông nhất của train. Kết quả chạy ngày 21/11/2026.

| Validation | Khoảng | Biểu diễn | C | Acc train | Acc val | Macro-F1 val |
|---|---:|---|---:|---:|---:|---:|
| 20% | 3 | đoán lớp đông nhất | | 0,334 | 0,329 | 0,165 |
| 20% | 3 | gốc | 3,16 | 0,411 | 0,405 | 0,396 |
| 20% | 3 | PCA k=11 | 0,001 | 0,378 | 0,380 | 0,375 |
| 20% | 4 | đoán lớp đông nhất | | 0,251 | 0,248 | 0,099 |
| 20% | 4 | gốc | 0,003 | 0,331 | 0,323 | 0,316 |
| 20% | 4 | PCA k=11 | 0,0003 | 0,306 | 0,310 | 0,306 |
| 30% | 3 | gốc | 0,003 | 0,413 | 0,407 | 0,399 |
| 40% | 3 | gốc | 0,003 | 0,411 | 0,406 | 0,396 |
| 30% | 4 | gốc | 0,003 | 0,333 | 0,322 | 0,316 |
| 40% | 4 | gốc | 0,01 | 0,332 | 0,323 | 0,317 |

Đủ 24 dòng trong `outputs/logistic/lr_summary.csv`; kết quả từng giá trị C trong
`lr_cv_results.csv`; biểu đồ `lr_tuning_curve_*`, `lr_confusion_*`, `lr_coefficients_*`.

Ma trận nhầm lẫn 3 khoảng, đặc trưng gốc, validation 20% (hàng là nhãn thật):

| | Dự đoán dễ | Dự đoán vừa | Dự đoán khó |
|---|---:|---:|---:|
| Dễ (< 1.154) | 633 | 254 | 430 |
| Vừa | 460 | 321 | 555 |
| Khó (> 1.699) | 354 | 329 | 664 |

Nhận xét:

- **Hơn ngẫu nhiên nhưng không nhiều.** Accuracy 3 khoảng 0,405 so với 0,333 khi
  đoán ngẫu nhiên; 4 khoảng 0,323 so với 0,25. Ổn định qua ba tỷ lệ chia (chênh
  dưới 0,003). Train và validation chỉ chênh khoảng 0,01, tức không overfit.
- **C không quan trọng.** Đường CV phẳng khi C ≥ 0,01 (macro-F1 0,395–0,396), nên
  việc C = 3,16 ở ô 3 khoảng và C ≈ 0,003 ở các ô khác chỉ là dao động trong vùng
  phẳng, không phải khác biệt thật. Giống Ridge: mô hình bị giới hạn bởi đặc trưng
  chứ không bởi regularization.
- **Khoảng giữa khó nhất.** Chỉ 24% bài mức vừa được đoán đúng; mô hình tuyến tính
  dồn dự đoán về hai đầu. Nhiều bài dễ bị đoán là khó (430/1.317), cho thấy cùng
  một cấu trúc bàn cờ có thể ứng với độ khó rất khác nhau.
- **Hệ số nhất quán với Ridge.** Hướng "khó hơn" (hệ số khoảng khó trừ khoảng dễ):
  `material_balance_to_move` +0,33, `fullmove` +0,28, `in_check` +0,25, `white_q`
  +0,21, `legal_moves` +0,19; hướng "dễ hơn": `black_attacked_squares` −0,28,
  `white_attacked_squares` −0,25. Đây cũng là các hệ số lớn nhất của Ridge.
- **PCA 1/3 số chiều làm giảm nhẹ** macro-F1 (0,396 → 0,375 với 3 khoảng), cùng
  chiều với Ridge.
- **So với Naive Bayes** (Thành viên 3, cùng nhãn, validation 20%): 3 khoảng gốc
  Logistic 0,396 so với NB 0,387 macro-F1. Ở 4 khoảng, Logistic giữ được 0,316
  trong khi NB trên đặc trưng gốc gần như chỉ đoán một lớp (0,102). Logistic không
  giả định các đặc trưng độc lập, nên không bị ảnh hưởng bởi tương quan mạnh giữa
  các đặc trưng đếm quân và ô bị tấn công.
