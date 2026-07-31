"""CANDI kit — a self-contained copy of the q19 dual-conditioning recipe.

Deliberately NOT a copy of sandbox/candi_v2/__init__.py: that module re-exports V2Decoder,
the model.py output heads and candi_loss.CANDI_LOSS, so a bare `import candi_kit` would drag
in the entire production loss stack. Only the training-path symbols are exported here.
"""

__version__ = "0.1.0"

from candi_kit.config import EncoderConfig
from candi_kit.encoder import V2Encoder
from candi_kit.model import RealDualCondModel, build_real_model, forward_full, nb_nll

__all__ = [
    "__version__",
    "EncoderConfig",
    "V2Encoder",
    "RealDualCondModel",
    "build_real_model",
    "forward_full",
    "nb_nll",
]
