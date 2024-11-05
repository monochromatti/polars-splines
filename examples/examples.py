import random

import polars as pl

import polars_splines  # noqa: F401


def linspace(start, stop, num):
    """Generates `num` evenly spaced values from `start` to `stop`."""
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


if __name__ == "__main__":
    # Generate linearly spaced values for x
    x = pl.Series("x", linspace(0.0, 1.0, 250))

    # Create the dataframe without numpy
    df = pl.concat(
        pl.DataFrame(
            dict(
                x=x,
                y1=x.map_elements(
                    lambda v: 10 * v + random.randint(0, 9) / 10,
                    return_dtype=pl.Float64,
                ).sin(),
                y2=x.map_elements(
                    lambda v: 11 * v - random.randint(0, 9), return_dtype=pl.Float64
                ).cos(),
                cat1=cat1val,
                cat2=cat2val,
            )
        )
        for cat1val in [chr(97 + i) for i in range(26)]
        for cat2val in [chr(97 + i) for i in range(26)]
    )
    print(df)

    # Define the apply_spline function
    def apply_spline(group, xi, value_vars, id_vars):
        id_vals = group.select(
            pl.repeat(pl.col(id_var).first(), len(xi)).alias(id_var) for id_var in id_vars
        )
        group = (
            group.select(
                pl.struct("x", col).spl.interpolate(xi=xi, fill_value=0.0).alias(col)
                for col in value_vars
            )
            .with_columns(xi, *id_vals)
        )
        return group

    # Generate the xi Series
    xi = pl.Series("x", linspace(0.3, 2, 1200))
    df_splines = df.group_by("cat1", "cat2").map_groups(
        lambda group: apply_spline(group, xi, ["y1", "y2"], ["cat1", "cat2"])
    )
    print(df_splines)
