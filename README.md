# Polars splines

An simple [extension plugin](https://github.com/pola-rs/pyo3-polars) for [py-polars](https://github.com/pola-rs/polars), that interfaces with the Rust cargo [splines](https://crates.io/crates/splines), for spline interpolation.

## Usage

The expression plugin adds `splines` to the expression namespace. This contains the method `spline` which acts on a `Series` of `Struct` type. The two fields corresponds to the (x, y) pairs to be interpolated. The `spline` method accepts a keyword argument `xi` for the interpolation points.

For example,

```python
import polars as pl
import polars_splines
import numpy as np

x = pl.Series("x", np.linspace(0, 1, 100))
y = pl.Series("y", np.sin(x))

df = pl.DataFrame({"x": x, "y": y})

xi = pl.Series("xi", np.linspace(0, 1, 1000))

dfi = df.select(
    pl.struct("x", "y").splines.spline(list(xi), fill_value=0.0).alias("yi")
).with_columns(xi)

```
