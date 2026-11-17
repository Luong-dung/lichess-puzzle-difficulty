import numpy as np
from sklearn.linear_model import Ridge

# Grid searched with GroupKFold on train in src/linear_analysis.py; 10**2.5 was chosen
# on the 20,000-row seed-42 sample (validation 20% and 40%; CV RMSE is flat below it).
ALPHAS = np.logspace(-3, 4, 15)


def build_model(seed=42, alpha=10 ** 2.5):
    return Ridge(alpha=alpha)
