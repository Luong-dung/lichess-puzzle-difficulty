# Thành viên 3: MLP, Phân loại Naive Bayes và Phân cụm GMM

## Chạy lại

```powershell
python -m src.mlp_analysis --sample 20000 --seed 42 --folds 5 --n-iter 15 --epochs 300  # -> outputs/mlp/
python -m src.classification --sample 20000 --seed 42                                 # -> outputs/classification/
python -m src.clustering_gmm --sample 20000 --seed 42 --n-bins 3 --k-min 2 --k-max 8   # -> outputs/clustering/
python -m unittest discover -s tests -v
```

Các mô hình đều dùng chung `load_sample`, `split_groups` và tiền xử lý với `src/run.py` (`src/preprocessing.py`). Toàn bộ việc chuẩn hóa, giảm chiều PCA và học ngưỡng phân vị đều chỉ fit trên tập train.

Kết quả dưới đây thu được từ lần chạy trên bộ đặc trưng v2 với 34 thuộc tính trích xuất từ thế cờ người giải (35 chiều sau khi mã hóa one-hot biến `turn`).

## Multi-Layer Perceptron (MLP)

Mạng nơ-ron được tinh chỉnh siêu tham số bằng RandomizedSearchCV với GroupKFold 5 fold theo ván cờ trên tập train. Không gian tìm kiếm gồm số lớp và nơ-ron (32 đến 128 nơ-ron, 1 đến 3 lớp ẩn), hệ số L2 `alpha` (10⁻⁴ đến 10⁻²), tốc độ học ban đầu `learning_rate_init` (10⁻³ đến 3×10⁻³) và kích thước batch (128, 256).

Cấu hình tốt nhất ở phân chia validation 20% là mạng 3 lớp ẩn (128, 64, 32), batch size 128, alpha 10⁻⁴ và tốc độ học ban đầu 10⁻³.

| Validation | Biểu diễn | Mô hình | RMSE train | RMSE val | MAE val | R² val |
|---|---|---|---:|---:|---:|---:|
| 20% | gốc | Trung bình train | 547,9 | 540,8 | 451,1 | 0,000 |
| 20% | gốc | OLS | 532,2 | 524,2 | 435,4 | 0,061 |
| 20% | gốc | MLP tuned | 472,7 | 507,4 | 415,5 | 0,120 |
| 20% | PCA k=11 | MLP tuned | 504,3 | 513,5 | 422,8 | 0,099 |
| 20% | SVD k=11 | MLP tuned | 505,5 | 514,6 | 424,4 | 0,095 |
| 30% | gốc | MLP tuned | 506,2 | 514,0 | 422,3 | 0,098 |
| 40% | gốc | MLP tuned | 501,2 | 517,9 | 423,5 | 0,093 |

Chi tiết toàn bộ 27 cấu hình nằm trong `outputs/mlp/mlp_summary.csv` và `outputs/mlp/mlp_summary.json`.

So với Ridge của Thành viên 1 (RMSE val 524,2 và R² 0,061), MLP trên không gian đặc trưng gốc giảm được 16,8 điểm RMSE và nâng R² lên gần gấp đôi (0,120). Khả năng mô hình hóa phi tuyến giúp mạng nơ-ron nắm bắt các tương tác phức tạp giữa bên tới lượt và các vị trí ô cờ bị tấn công.

Khi giảm chiều xuống 11 thành phần (1/3 số chiều ban đầu), kết quả giữa PCA và TruncatedSVD tiếp tục gần như tương đương nhau do dữ liệu đã được chuẩn hóa trước. Tuy nhiên, cả hai đều làm giảm R² từ 0,120 xuống khoảng 0,095 đến 0,099 vì đánh mất một phần tín hiệu phương sai nhỏ nhưng hữu ích.

Phần dư của MLP phân bố theo mức Rating (`outputs/mlp/mlp_residuals_by_rating.csv`):
- Nhóm Rating thấp nhất (399 đến 790): dự đoán cao hơn thực tế trung bình 655 điểm.
- Nhóm Rating trung bình (1.422 đến 1.599): sai lệch rất nhỏ, phần dư trung bình chỉ -26,5 điểm và MAE là 183,4.
- Nhóm Rating cao nhất (2.208 đến 3.065): dự đoán thấp hơn thực tế trung bình 836 điểm.

## Phân loại Naive Bayes

Bài toán phân loại chia Rating thành 3 hoặc 4 khoảng bằng nhau theo phân vị. Nhằm tuân thủ nghiêm ngặt nguyên tắc chống rò rỉ dữ liệu, toàn bộ các ngưỡng cắt phân vị đều được tính toán duy nhất trên tập train thông qua hàm `fit_rating_bins`, sau đó áp dụng sang tập validation bằng `apply_rating_bins`.

Ở tỷ lệ validation 20%:
- Ngưỡng 3 khoảng: [1154,0; 1699,0], tương ứng nhãn 0 (< 1154), nhãn 1 (1154 đến 1699) và nhãn 2 (> 1699).
- Ngưỡng 4 khoảng: [1036,0; 1409,0; 1865,0].

Mô hình Gaussian Naive Bayes được đánh giá trên cả đặc trưng gốc và đặc trưng giảm chiều PCA (k = 11):

| Số khoảng | Biểu diễn | Validation 20%: Acc | Val 20%: Macro F1 | Val 30%: Acc | Val 30%: Macro F1 | Val 40%: Acc | Val 40%: Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 3 khoảng | gốc | 0,390 | 0,387 | 0,396 | 0,383 | 0,397 | 0,386 |
| 3 khoảng | PCA k=11 | 0,362 | 0,317 | 0,346 | 0,266 | 0,349 | 0,272 |
| 4 khoảng | gốc | 0,248 | 0,102 | 0,251 | 0,104 | 0,250 | 0,103 |
| 4 khoảng | PCA k=11 | 0,280 | 0,223 | 0,271 | 0,203 | 0,274 | 0,206 |

Chi tiết các ma trận nhầm lẫn nằm tại `outputs/classification/nb_confusion_*.png`.

Với 3 khoảng độ khó, dự đoán ngẫu nhiên tương ứng 33,3%. Naive Bayes trên đặc trưng gốc đạt độ chính xác khoảng 39,0% đến 39,7% và Macro F1 đạt 0,387, nhỉnh hơn ngẫu nhiên khoảng 6%. 

Với 4 khoảng độ khó, ngẫu nhiên là 25,0%. Trên đặc trưng gốc, Naive Bayes gặp hiện tượng sụp đổ dự đoán vào các lớp phổ biến, khiến Macro F1 chỉ đạt 0,102. Khi áp dụng PCA để trực giao hóa các đặc trưng, giả định độc lập có điều kiện của Naive Bayes được cải thiện đáng kể, giúp Macro F1 tăng từ 0,102 lên 0,223.

## Phân cụm Gaussian Mixture Models (GMM)

Mô hình GMM khám phá cấu trúc cụm hình thái thế cờ thông qua phân phối hỗn hợp Gauss. Dữ liệu được tiền xử lý và fit trên tập train của phép chia 20%.

Quá trình quét tham số thực hiện trên 28 cấu hình gồm số thành phần k từ 2 đến 8 và 4 dạng ma trận hiệp phương sai (`full`, `tied`, `diag`, `spherical`). Tiêu chí chọn mô hình dựa trên điểm cực tiểu của đường cong BIC (`outputs/clustering/gmm_bic_aic.png`):

| k | BIC (full) | BIC (tied) | BIC (diag) | BIC (spherical) |
|---:|---:|---:|---:|---:|
| 2 | -809.514 | 132.667 | 462.250 | 1.435.521 |
| 3 | -831.914 | 112.693 | 382.184 | 1.369.112 |
| 4 | -1.123.884 | 109.527 | 277.347 | 1.330.059 |
| 5 | -1.378.862 | -104.882 | -160.307 | 1.310.276 |
| 6 | -1.483.203 | -130.688 | -323.212 | 1.290.543 |
| 7 | -1.524.882 | -334.147 | -468.682 | 1.277.983 |
| 8 | -1.568.550 | -211.638 | -516.222 | 1.259.046 |

- Ma trận hiệp phương sai dạng `full` vượt trội hoàn toàn so với các dạng còn lại. Giá trị BIC giảm liên tục khi tăng số cụm, phản ánh việc thế cờ cờ vua có tương quan nội tại rất chặt chẽ giữa các loại quân và vị trí kiểm soát.
- Cấu hình tối ưu được chọn là k = 8 với dạng `full` (BIC đạt -1.568.550).
- Hệ số Silhouette đo được trên toàn bộ không gian là 0,071 trên train và 0,070 trên validation. Giá trị gần 0 cho thấy các thế cờ tạo thành một đám mây điểm phân bố liên tục, không có ranh giới phân tách cụm rõ rệt.
- Đối chiếu với nhãn khoảng Rating: chỉ số ARI đạt 0,0018 và NMI đạt 0,0041. Tỷ lệ phương sai Rating được giải thích bởi các cụm (eta squared) chỉ đạt 0,0098 (khoảng 1%). Điều này chứng minh GMM phân nhóm thế cờ theo cấu trúc bàn cờ (tàn cuộc ít quân, trung cuộc phức tạp, trạng thái vua) chứ không phân tách theo trình độ hay độ khó bài toán.

## Nhận xét

- **MLP vượt trội hơn mô hình tuyến tính nhưng vẫn underfit.** Với R² đạt 0,120, MLP giải thích được phương sai tốt gấp hai lần Ridge (0,061). Mạng nơ-ron học được các mối liên hệ phi tuyến mà OLS và Ridge bỏ sót. Tuy nhiên, mức R² 0,120 vẫn tương đối khiêm tốn. Khoảng cách RMSE giữa train và validation ở split 20% là 34,7 điểm (472,7 so với 507,4), và ở split 30% chỉ còn 7,8 điểm. Đường học (`mlp_learning_curve.png`) đi ngang từ khoảng 6.000 dòng train, khẳng định giới hạn chính nằm ở thông tin của bộ đặc trưng tĩnh chứ không phải do thiếu dữ liệu hay do overfit.
- **Phần dư co cụm mạnh về giá trị trung vị.** Tương tự như Ridge, MLP có xu hướng kéo các dự đoán về vùng an toàn ở giữa phân bố (`mlp_residuals.png`). Các bài tập nhập môn bị đoán khó hơn thực tế khoảng 655 điểm, trong khi bài tập đại kiện tướng bị đoán thấp hơn khoảng 836 điểm. Dù vậy, biên độ lệch này đã được thu hẹp đáng kể so với mức lệch 762 và 936 điểm của Ridge.
- **Tác động trái ngược của PCA giữa hồi quy và phân loại.** Với MLP, việc nén còn 11 chiều làm giảm R² từ 0,120 xuống 0,099 do mất mát thông tin cục bộ. Ngược lại, với Naive Bayes ở bài toán 4 lớp, PCA đóng vai trò bộ lọc khử tương quan, giúp giải tỏa vi phạm giả định độc lập và đưa Macro F1 từ 0,102 lên 0,223.
- **Bản chất của phân cụm thế cờ.** Kết quả từ GMM với ARI và NMI xấp xỉ 0 chỉ ra rằng phân cụm không giám sát không phát hiện được độ khó của puzzle. Các cụm thế cờ đại diện cho các giai đoạn và dạng hình học bàn cờ khác nhau (`gmm_clusters_pc.png`, `gmm_rating_by_cluster.png`). Hai thế cờ có cùng lượng quân và cùng cấu trúc có thể chênh lệch hàng nghìn điểm Rating chỉ vì một bên có nước đi chiến thuật ẩn mà các thuộc tính đếm quân tĩnh không thể hiện được.
