use polars::error::polars_err;
use polars::prelude::{DataType, NamedFrom, PolarsError, PolarsResult, Series};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use splines::{Interpolation, Key, Spline};

#[derive(Deserialize)]
struct SplineKwargs {
    fill_value: Option<f64>,
    xi: Vec<f64>,
    method: Option<String>,
}

#[polars_expr(output_type=Float64)]
fn spline(inputs: &[Series], kwargs: SplineKwargs) -> PolarsResult<Series> {
    if inputs.len() != 1 {
        return Err(PolarsError::InvalidOperation(
            "`spline` only works on a single struct".into(),
        ));
    }
    let input = &inputs[0];
    match input.dtype() {
        DataType::Struct(_) => {
            let fields = input.struct_()?.fields();

            let x: Vec<f64> = fields[0].f64()?.into_iter().flatten().collect();
            let y: Vec<f64> = fields[1].f64()?.into_iter().flatten().collect();

            let interpolator = match kwargs.method.as_deref() {
                Some("linear") => Ok(Interpolation::Linear),
                Some("cosine") => Ok(Interpolation::Cosine),
                Some("catmullrom") => Ok(Interpolation::CatmullRom),
                None => Err(PolarsError::InvalidOperation(
                    "Method not specified. Please specify a valid method: 'linear', 'cosine', or 'catmullrom'.".into(),
                )),
                Some(other) => Err(PolarsError::InvalidOperation(format!(
                    "Invalid method: {}. Valid methods are: 'linear', 'cosine', and 'catmullrom'.",
                    other
                ).into())),
            }?;

            let keys: Vec<Key<f64, f64>> = x
                .into_iter()
                .zip(y.into_iter())
                .map(|(x, y)| Key::new(x, y, interpolator))
                .collect();

            let spline = Spline::from_vec(keys);
            let yi: Vec<Option<f64>> = match kwargs.fill_value {
                Some(fill_val) => kwargs
                    .xi
                    .iter()
                    .map(|&xi_val| Some(spline.sample(xi_val).map_or(fill_val, |v| v)))
                    .collect(),
                None => kwargs
                    .xi
                    .iter()
                    .map(|&xi_val| spline.sample(xi_val))
                    .collect(),
            };

            Ok(Series::new(fields[1].name(), yi)
                .cast(fields[1].dtype())
                .map_err(|e| {
                    PolarsError::ComputeError(format!("Failed to cast Series: {}", e).into())
                })?)
        }
        _ => Err(PolarsError::InvalidOperation(
            "`spline` only works on a struct (pl.struct(col(\"x\"), col(\"y\")))".into(),
        )),
    }
}
