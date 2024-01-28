# Polars splines

A simple [extension plugin](https://github.com/pola-rs/pyo3-polars) for [polars](https://github.com/pola-rs/polars) Python API that uses the Rust cargo [splines](https://crates.io/crates/splines) for spline interpolation.

## Installation
Install with `pip` or `uv` from [PyPI](https://pypi.org/project/polars-splines/), or build from source with [`maturin`](https://github.com/PyO3/maturin).

## Usage
The expression plugin adds `spl` to the expression namespace. This contains the method `spline` which acts on columns of `Struct` type. The two fields corresponds to the (x, y) pairs to be interpolated. The `interpolate` method accepts a keyword argument `xi` for the interpolation points.

For example,

```python
import polars as pl
import polars_splines
import numpy as np

x = pl.Series("x", np.linspace(0, 1, 100))
y = pl.Series("y", np.sin(x))

df = pl.DataFrame({"x": x, "y": y})

xi = pl.Series("xi", np.linspace(0, 1, 1000))

df = df.with_columns(
    pl.lit(xi).alias("xi"),
    pl.struct("x", "y").spl.interpolate(xi, fill_value=0.0, method="catmullrom").alias("yi")
)

```
