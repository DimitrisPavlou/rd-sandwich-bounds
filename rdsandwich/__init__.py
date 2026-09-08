"""
rdsandwich — a PyTorch re-implementation of

    Towards Empirical Sandwich Bounds on the Rate-Distortion Function
    Yibo Yang, Stephan Mandt. ICLR 2022. https://arxiv.org/abs/2111.12166

This package ports the original TensorFlow / tensorflow-compression codebase
(``RD-sandwich-master``) to PyTorch, and reorganizes it into an installable,
importable library (rather than a folder of standalone scripts) so that the
three core algorithms in the paper —

    1. the R-D upper bound (a beta-VAE trained with an SGD version of the
       Blahut-Arimoto algorithm; see ``rdsandwich.upper_bound``),
    2. the R-D lower bound (Csiszár's dual characterization, optimized with
       a global-maximization inner loop; see ``rdsandwich.lower_bound``), and
    3. the classical Blahut-Arimoto algorithm on a discretized source
       (``rdsandwich.utils.ba``),

— can be reused, tested, and scaled independently of any one experiment.

See the top-level README.md for the mapping from the original TF scripts to
this package, and ``experiments/`` for shell scripts that reproduce the
paper's commands using the new CLIs in ``scripts/``.
"""

__version__ = "0.1.0"
