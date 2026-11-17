# Phân công nhóm 3 người

| Thành viên | Module phụ trách | Sản phẩm |
|---|---|---|
| 1 Nguyễn Quốc Doanh | `src/models/linear.py`, EDA và PCA | Ridge/Linear; biểu đồ dữ liệu; phân tích phương sai PCA |
| 2 Lường Tiến Dũng | `src/models/random_forest.py`, SVD và K-Means | Random Forest; thử n_estimators/max_depth/min_samples_leaf/max_features; phân cụm |
| 3 Trần Bảo Ngọc | `src/models/mlp.py`, GMM và phân loại bổ sung | MLP; GMM; điều phối Naive Bayes và Logistic Regression |

Cả nhóm cùng rà soát `data.py`, `features.py`, `evaluation.py`, thống nhất dữ liệu và seed.
Mỗi người viết lý thuyết, kết quả, nhận xét và trình bày mô hình của mình.
Không đổi cách chia dữ liệu riêng cho từng mô hình. Nhánh công việc gợi ý:
`member-1/linear`, `member-2/random-forest`, `member-3/mlp`; tạo pull request vào `main`.

## Phân công chi tiết phần còn lại

Thành viên 1 đã xong EDA, PCA và Ridge (`docs/THANH_VIEN_1.md`), nên nhận thêm
phần dùng chung: bổ sung đặc trưng, Logistic Regression (cùng họ tuyến tính),
bảng so sánh cuối và ghép báo cáo. Ngọc vẫn điều phối phần phân loại: định nghĩa
nhãn và Naive Bayes; Doanh làm Logistic trên cùng nhãn đó.

| Người | Việc | Đầu ra |
|---|---|---|
| **Doanh** | Thêm đặc trưng python-chess vào `src/features.py` (số nước hợp lệ, đang bị chiếu, quân bị tấn công/không được bảo vệ, an toàn vua, chênh lệch vật chất theo bên tới lượt). Giữ ≥ 15 thuộc tính, cập nhật test | PR đặc trưng, danh sách đặc trưng trong `docs/DU_LIEU.md` |
| | Chạy lại EDA, PCA, Ridge với bộ đặc trưng mới | Cập nhật `docs/THANH_VIEN_1.md` |
| | Logistic Regression trên nhãn của Ngọc, đặc trưng gốc và PCA 1/3 số chiều | Bảng accuracy, macro-F1, ma trận nhầm lẫn |
| | Script gom `outputs/` thành bảng so sánh cuối; ghép báo cáo | `src/compare.py`, file báo cáo |
| **Dũng** | Tuning Random Forest bằng GroupKFold trên train (n_estimators, max_depth, min_samples_leaf, max_features); learning curve, phần dư, feature importance | `outputs/random_forest/`, `docs/THANH_VIEN_2.md` |
| | So sánh SVD với PCA (dùng kết quả SVD chưa tâm hóa của Doanh làm điểm xuất phát) | Mục SVD trong báo cáo |
| | K-Means: chọn k (elbow, silhouette), trực quan hóa trên PC1–PC2, đối chiếu Rating từng cụm | `outputs/clustering/kmeans_*` |
| | Hướng dẫn tái lập toàn bộ trong `README.md` | README hoàn chỉnh |
| **Ngọc** | Tuning MLP bằng GroupKFold trên train (số lớp, số nơ-ron, alpha, learning rate); đường loss, overfit, phần dư | `outputs/mlp/`, `docs/THANH_VIEN_3.md` |
| | GMM: chọn số thành phần (BIC/AIC), loại covariance; so sánh với K-Means bằng cùng độ đo | `outputs/clustering/gmm_*` |
| | Chia Rating thành 3–4 khoảng theo phân vị **học từ train**; Naive Bayes trên đặc trưng gốc và 1/3 số chiều | `src/classification.py`, nhãn dùng chung cho Logistic |
| | Slide trình bày | File slide |

Thống nhất chung: seed 42, mẫu 20.000, phép chia theo GameUrl như `src/run.py`;
tuning chỉ trên train bằng GroupKFold 5 fold; K-Means và GMM dùng cùng độ đo
(silhouette, ARI/NMI với nhãn khoảng Rating của Ngọc).

## Lịch 2 tuần (16/11 – 29/11/2026)

Mốc khóa quan trọng nhất: **bộ đặc trưng khóa ngày 18/11**. Trước đó Dũng và Ngọc
viết code tuning/phân cụm trên đặc trưng hiện tại; sau mốc này chỉ chạy lại, không
đổi đặc trưng nữa.

### Tuần 1: hoàn thiện code (16/11 – 22/11)

| Ngày | Doanh | Dũng | Ngọc |
|---|---|---|---|
| T2 16/11 | Họp khởi động; mở PR `member-1/linear` vào `main` | Review PR của Doanh; tạo nhánh `member-2/random-forest` | Review PR của Doanh; tạo nhánh `member-3/mlp` |
| T3 17/11 | Viết đặc trưng python-chess + test | Code tuning RF với GroupKFold | Code tuning MLP với GroupKFold |
| T4 18/11 | **Merge đặc trưng mới vào `main`, báo cả nhóm** | Rebase lên `main` | Rebase lên `main`; code chia nhãn Rating |
| T5 19/11 | Chạy lại EDA, PCA, Ridge | Chạy tuning RF; phần dư, feature importance | Chạy tuning MLP; Naive Bayes |
| T6 20/11 | Logistic Regression (gốc và 1/3 chiều) | So sánh SVD và PCA | GMM (BIC/AIC) |
| T7 21/11 | Viết `src/compare.py` | K-Means, đối chiếu Rating | So sánh GMM với K-Means |
| CN 22/11 | **Họp tuần 1**: mỗi người trình bày kết quả 10 phút; merge PR của Dũng và Ngọc | | |

Hết tuần 1: mọi code nằm trên `main`, `python -m unittest` chạy qua.

### Tuần 2: báo cáo và slide (23/11 – 29/11)

| Ngày | Doanh | Dũng | Ngọc |
|---|---|---|---|
| T2 23/11 | Chạy toàn bộ trên `main`, xuất bảng so sánh cuối | Sửa lỗi phát sinh khi chạy chung | Sửa lỗi phát sinh khi chạy chung |
| T3 24/11 | Viết: giới thiệu, dữ liệu, tiền xử lý | Viết: lý thuyết + kết quả RF | Viết: lý thuyết + kết quả MLP |
| T4 25/11 | Viết: EDA, PCA, Ridge, Logistic | Viết: SVD, K-Means | Viết: GMM, phân loại, Naive Bayes |
| T5 26/11 | **Ghép báo cáo**, viết so sánh và kết luận | Hoàn thiện README tái lập; thử tái lập trên máy sạch | Làm slide (mỗi người gửi 3–4 slide phần mình trước 12h) |
| T6 27/11 | Review chéo phần của Dũng | Review chéo phần của Ngọc | Review chéo phần của Doanh |
| T7 28/11 | Sửa theo review; chốt 15–20 trang | Tập thuyết trình | Tập thuyết trình; chốt slide |
| CN 29/11 | **Dự phòng và nộp** (trưởng nhóm nộp) | | |

Phân bổ báo cáo (15–20 trang): Doanh 7–8 trang (mở đầu, dữ liệu, EDA, PCA, Ridge,
Logistic, so sánh, kết luận); Dũng 5–6 trang; Ngọc 5–6 trang.

### Quy tắc phối hợp

- Mỗi việc làm trên nhánh riêng, PR vào `main`, ít nhất một người khác review.
- Kết quả nào đưa vào báo cáo phải chạy lại được bằng một lệnh ghi trong README.
- Báo tiến độ ngắn trong nhóm chat mỗi tối; bị chặn quá nửa ngày thì báo ngay.
- Trễ một mốc thì dùng ngày dự phòng 29/11, không lùi mốc khóa đặc trưng 18/11.

## Checklist yêu cầu

- [x] Khung trích xuất >15 đặc trưng số và phân loại từ FEN.
- [x] Ba module hồi quy và runner chung; chia theo ván gốc để hạn chế rò rỉ.
- [x] Runner có tỷ lệ validation 20%, 30%, 40%; dữ liệu gốc, PCA, SVD.
- [x] Xác thực FEN bằng python-chess; thống kê thiếu, lỗi, trùng thế cờ.
- [x] EDA, biểu đồ 4–6 thành phần, phương sai giải thích và so sánh hai cách giảm chiều.
- [x] Bổ sung đặc trưng python-chess (34 thuộc tính, khóa 18/11); chạy lại EDA, PCA, Ridge (Doanh, xong 17/11).
- [ ] K-Means và GMM; đánh giá định lượng, trực quan hóa và đối chiếu Rating (Dũng, Ngọc, 20–21/11).
- [ ] Điều chỉnh siêu tham số trên train; phân tích overfit và phần dư (Ridge xong; RF, MLP 19/11).
- [ ] Phân loại bổ sung: chia Rating thành 3–4 khoảng gần cân bằng; ngưỡng học từ train (Ngọc, 18/11).
- [ ] Naive Bayes và Logistic trên đặc trưng gốc và giảm còn 1/3 số chiều (Ngọc, Doanh, 19–20/11).
- [ ] Báo cáo 15–20 trang, slide, hướng dẫn tái lập và bảng so sánh cuối cùng (cả nhóm, 23–28/11).

## Mốc

- Đăng ký: 23:59 ngày 31/10/2026; chỉ trưởng nhóm đăng ký.
- Giữa kỳ: tiền xử lý, giảm chiều, trực quan hóa và ít nhất hai mô hình hoàn tất.
- Khóa đặc trưng: 18/11/2026. Code hoàn tất: 22/11/2026. Nộp: 29/11/2026.
- Đã xác nhận với giảng viên: cần ít nhất 15 thuộc tính sau trích xuất từ FEN.
