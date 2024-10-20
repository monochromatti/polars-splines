import numpy as np
import polars as pl
import polars_splines  # noqa: F401

if __name__ == "__main__":
    x = pl.Series("x", np.linspace(0.0, 1.0, 250))

    df = pl.concat(
        pl.DataFrame(
            dict(
                x=x,
                y1=np.sin(10 * x + np.random.randint(10) / 10),
                y2=np.cos(11 * x - np.random.randint(10)),
                cat1=cat1val,
                cat2=cat2val,
            )
        )
        for cat1val in [chr(97 + i) for i in range(26)]
        for cat2val in [chr(97 + i) for i in range(26)]
    )
    print(df)

    def apply_spline(group, xi, value_vars, id_vars):
        id_vals = group.select(pl.col(*id_vars).first())
        group = group.select(
            pl.struct("x", col).spl.interpolate(xi=xi, fill_value=0.0).alias(col)
            for col in value_vars
        ).with_columns(xi, *id_vals)
        return group

    xi = pl.Series("x", np.linspace(0.3, 2, 1200))
    df_splines = df.group_by(["cat1", "cat2"]).map_groups(
        lambda group: apply_spline(group, xi, ["y1", "y2"], ["cat1", "cat2"])
    )
    print(df_splines)
