"""Generic neural-network building blocks shared across the bound algorithms.

This subpackage holds only reusable *network* classes (MLPs, conv nets, GDN,
normalizing flows). The bound-specific models that also compute an objective
(e.g. the beta-VAE ``RDUBModel``) live in ``rdsandwich.upper_bound`` /
``rdsandwich.lower_bound`` instead.
"""
from .mlp import GDN, get_activation, make_mlp
from .conv import get_convnet
from .flows import MADE, MAF, MAFLayer

__all__ = [
    "GDN", "get_activation", "make_mlp", "get_convnet", "MADE", "MAF", "MAFLayer",
]
