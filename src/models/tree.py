from sklearn.tree import DecisionTreeRegressor


def build_model(seed=42):
    return DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=seed)
