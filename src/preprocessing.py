"""Feature preprocessing identical to src/run.py, fitted only on training rows."""
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocess(numeric, center=True):
    return ColumnTransformer([
        ("numeric", StandardScaler(with_mean=center), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["turn"])])


def reduced_dimensions(numeric):
    return max(1, (len(numeric) + 2) // 3)


def build_regressor(numeric, estimator, representation="original", seed=42):
    steps = [("preprocess", build_preprocess(numeric))]
    if representation == "pca":
        steps.append(("reduce", PCA(n_components=reduced_dimensions(numeric), random_state=seed)))
    elif representation == "svd":
        steps.append(("reduce", TruncatedSVD(n_components=reduced_dimensions(numeric), random_state=seed)))
    steps.append(("model", estimator))
    return TransformedTargetRegressor(regressor=Pipeline(steps), transformer=StandardScaler())


def clean_names(names):
    return [name.split("__", 1)[-1] for name in names]
