from sklearn.neural_network import MLPRegressor

# Không gian tìm kiếm (tuning)
DEFAULT_PARAM_GRID = {
    "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],   # Số lớp ẩn và số neuron mỗi lớp
    "alpha": [1e-4, 1e-3, 1e-2],    # Tham số regularization (L2 penalty)
    "learning_rate_init": [1e-3, 3e-3],     # Tốc độ học ban đầu
    "batch_size": [128, 256],   # Kích thước batch
}


def build_model(
    seed=42,
    hidden_layer_sizes=(64, 32),
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    **kwargs
):
    # Khởi tạo mô hình hồi quy MLPRegressor với các tham số mặc định
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,      # Dừng sớm nếu không cải thiện trên tập validation
        random_state=seed,
        **kwargs
    )