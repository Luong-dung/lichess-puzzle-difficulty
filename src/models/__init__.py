from .linear import build_model as linear_model
from .random_forest import build_model as random_forest_model
from .mlp import build_model as mlp_model

MODELS = {"ridge": linear_model, "random_forest": random_forest_model, "mlp": mlp_model}
