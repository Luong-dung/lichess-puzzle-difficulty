from sklearn.neural_network import MLPRegressor


def build_model(seed=42):
    return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                        early_stopping=True, random_state=seed)
