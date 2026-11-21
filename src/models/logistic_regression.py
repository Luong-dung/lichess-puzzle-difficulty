import numpy as np
from sklearn.linear_model import LogisticRegression

# C = 1/lambda (L2). Chọn bằng GroupKFold trên train trong src/logistic_analysis.py.
C_GRID = np.logspace(-4, 2, 13)


def build_model(seed=42, C=1.0, class_weight=None):
    return LogisticRegression(
        C=C,
        solver="lbfgs",        # softmax đa lớp
        max_iter=2000,
        class_weight=class_weight,
        random_state=seed,
    )
