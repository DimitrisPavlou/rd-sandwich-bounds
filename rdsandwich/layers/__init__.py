"""Generic neural-network building blocks shared across the bound algorithms.

This subpackage holds only reusable *network* classes (MLPs, conv nets, GDN,
normalizing flows). The trainable models built from them (e.g. the beta-VAEs
that compute an R-D objective) live in ``rdsandwich.models``.
"""
from .gdn import GDN, NonNegativeParameterizer
from .mlp import get_activation, make_mlp
from .conv import get_convnet
from .flows import MADE, MAF, MAFLayer
from .deep_factorized import DeepFactorized
from .channelwise_ar import ChannelwiseARTransform

__all__ = [
    "GDN", "NonNegativeParameterizer", "get_activation", "make_mlp", "get_convnet",
    "MADE", "MAF", "MAFLayer", "DeepFactorized", "ChannelwiseARTransform",
]
