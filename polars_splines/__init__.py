import polars as pl
from polars.utils.udfs import _get_shared_lib_location

# Boilerplate needed to inform Polars of the location of binary wheel.
lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("splines")
class SplinesNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def spline(self, xi=None, method="linear", fill_value=None) -> pl.Expr:
        if xi is None:
            raise ValueError("Interpolation points `xi` must be provided")
        return self._expr._register_plugin(
            lib=lib,
            symbol="spline",
            is_elementwise=True,
            kwargs={"xi": xi, "method": method, "fill_value": fill_value},
        )
