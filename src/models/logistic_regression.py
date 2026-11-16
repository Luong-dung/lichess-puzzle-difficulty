from sklearn.linear_model import LogisticRegression


def build_model(seed=42):
    return LogisticRegression(
        C=1.0,                 # nghịch đảo hệ số L2
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,         # đủ lớn để tránh cảnh báo không hội tụ
        class_weight=None,     # đổi thành "balanced" nếu các khoảng lệch nhau
        random_state=seed,
    )