from sklearn.linear_model import Ridge


def build_model(seed=42):
    return Ridge(alpha=10.0)
