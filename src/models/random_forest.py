from sklearn.ensemble import RandomForestRegressor


def build_model(seed=42):
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=5,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )
