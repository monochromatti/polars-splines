import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path


@pl.api.register_expr_namespace("splines")
class SplinesNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def spline(
        self, xi: list[float], method="linear", fill_value: float = None
    ) -> pl.Expr:
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="spline",
            args=self._expr,
            is_elementwise=False,
            kwargs={"xi": xi, "method": method, "fill_value": fill_value},
        )
