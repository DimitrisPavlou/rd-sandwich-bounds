"""R-D lower-bound models: networks mapping ``x`` to a scalar ``log u(x)``."""
from .log_u import build_log_u_model

__all__ = ["build_log_u_model"]
