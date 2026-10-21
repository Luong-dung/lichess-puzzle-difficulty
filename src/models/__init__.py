from .linear import build_model as linear_model
from .tree import build_model as tree_model
from .mlp import build_model as mlp_model

MODELS = {"ridge": linear_model, "tree": tree_model, "mlp": mlp_model}
