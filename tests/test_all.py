import math
import random

import polars as pl
import pytest
from polars import col

import polars_splines

x = [float(i) for i in range(100)]
y = [math.cos(i / 20) + 2 * random.random() for i in x]


@pytest.mark.parametrize("method", ["linear", "cosine", "catmullrom"])
def test_closeness(method):
    df = pl.DataFrame({"x": x, "y": y})
    s = df.select(
        (pl.struct("x", "y").spl.interpolate(x, method=method) - col("y"))
        .abs()
        .alias("diff")
    ).to_series()

    assert (
        s < 1e-6
    ).all(), "Expected interpolated values to be within 1e-6 from original"


@pytest.mark.parametrize("method", ["linear", "cosine", "catmullrom"])
def test_nonmonotonic(method):
    df = pl.DataFrame({"x": x, "y": y})
    xi = [float(i) for i in range(40)]
    xi += xi[::-1]
    df.select(
        pl.struct("x", "y").spl.interpolate(xi, method=method).alias("yi")
    ).to_series()


@pytest.mark.parametrize("method", ["linear", "cosine", "catmullrom"])
def test_out_of_bounds(method):
    df = pl.DataFrame({"x": x, "y": y})
    xi = [float(i) for i in range(200)]
    s = df.select(
        pl.struct("x", "y").spl.interpolate(xi, method=method).alias("yi")
    ).to_series()
    assert s.is_null().sum() > 100, "Expected nulls for all out of bounds values"
