"""Generic neural-network building blocks shared across the bound algorithms.

This subpackage holds only reusable *network* classes (MLPs, conv nets, GDN,
normalizing flows). The bound-specific models that also compute an objective
(e.g. the beta-VAE ``RDUBModel``) live in ``rdsandwich.upper_bound`` /
``rdsandwich.lower_bound`` instead.
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
