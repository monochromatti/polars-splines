# polars-splines: An expression plugin for Polars

[![Documentation](https://img.shields.io/badge/Documentation-black?logo=readthedocs)](https://monochromatti.github.io/polars-splines/)

A simple [extension plugin](https://github.com/pola-rs/pyo3-polars) for [polars](https://github.com/pola-rs/polars) Python API that uses the Rust cargo [splines](https://crates.io/crates/splines) for spline interpolation.

## Installation
Install with `pip` or `uv` from [PyPI](https://pypi.org/project/polars-splines/), or build from source with [`maturin`](https://github.com/PyO3/maturin).

## Minimal example

```python
import polars as pl
import polars_splines # Adds `spl` to expression namespace

df = pl.DataFrame({"x": [0, 1, 2], "y": [3, 1, 5]})

xi = pl.Series("xi", [0, 0.5, 1.7])

dfi = df.select(
    pl.struct("x", "y").spl.interpolate(xi).alias("yi")
)
```
