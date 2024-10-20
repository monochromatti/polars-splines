import polars as pl
from polars.plugins import register_plugin_function
from typing import Iterable
from pathlib import Path


@pl.api.register_expr_namespace("spl")
class SplinesNamespace:
    implemented_methods = {"linear", "cosine", "catmullrom"}

    def __init__(self, expr: pl.Expr):
        self.expr = expr

    def interpolate(
        self,
        xi: Iterable[int | float] | float,
        method="linear",
        fill_value: float = None,
    ) -> pl.Expr:
        return register_plugin_function(
            function_name="interpolate",
            plugin_path=Path(__file__).parent,
            args=self.expr,
            is_elementwise=False,
            returns_scalar = False,
            kwargs={
                "xi": self._cast_xi(xi),
                "method": self._verify_method(method),
                "fill_value": fill_value,
            },
        )

    def _cast_xi(self, xi: Iterable[int | float] | float) -> list:
        if isinstance(xi, float):
            xi = [xi]
        elif not isinstance(xi, list):
            xi = list(xi)

    def _verify_method(self, method: str) -> None:
        if method not in self.implemented_methods:
            raise ValueError(
                f"Method {method} is not implemented. Choose one of {self.implemented_methods}"
            )
