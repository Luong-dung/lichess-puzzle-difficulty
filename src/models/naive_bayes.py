from sklearn.naive_bayes import GaussianNB


def build_model(seed=42):
    return GaussianNB(var_smoothing=1e-9)