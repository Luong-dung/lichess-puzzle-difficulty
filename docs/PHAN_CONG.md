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

## Checklist yêu cầu

- [x] Khung trích xuất >15 đặc trưng số và phân loại từ FEN.
- [x] Ba module hồi quy và runner chung; chia theo ván gốc để hạn chế rò rỉ.
- [x] Runner có tỷ lệ validation 20%, 30%, 40%; dữ liệu gốc, PCA, SVD.
- [ ] Xác thực FEN bằng python-chess; thống kê thiếu, lỗi, trùng thế cờ.
- [ ] EDA, biểu đồ 4–6 thành phần, phương sai giải thích và so sánh hai cách giảm chiều.
- [ ] K-Means và GMM; đánh giá định lượng, trực quan hóa và đối chiếu Rating.
- [ ] Điều chỉnh siêu tham số trên train; phân tích overfit và phần dư.
- [ ] Phân loại bổ sung: chia Rating thành 3–4 khoảng gần cân bằng; ngưỡng học từ train.
- [ ] Naive Bayes và Logistic trên đặc trưng gốc và giảm còn 1/3 số chiều.
- [ ] Báo cáo 15–20 trang, slide, hướng dẫn tái lập và bảng so sánh cuối cùng.

## Mốc

- Đăng ký: 23:59 ngày 31/10/2026; chỉ trưởng nhóm đăng ký.
- Giữa kỳ: tiền xử lý, giảm chiều, trực quan hóa và ít nhất hai mô hình hoàn tất.
- Cần xác nhận với giảng viên về việc tính 15 thuộc tính sau trích xuất từ FEN.
