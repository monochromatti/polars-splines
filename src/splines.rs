use polars::error::polars_err;
use polars::prelude::{DataType, NamedFrom, PolarsError, PolarsResult, Series};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use splines::{Interpolation, Key, Spline};

#[derive(Deserialize)]
struct SplineKwargs {
    xi: Vec<f64>,
    method: Option<String>,
    fill_value: Option<f64>,
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

            let interpolator = get_method(kwargs.method.as_deref())?;

            let keys: Vec<Key<f64, f64>> = x
                .into_iter()
                .zip(y.into_iter())
                .map(|(x, y)| Key::new(x, y, interpolator))
                .collect();
            let spline = Spline::from_vec(keys);

            let xi_iter = kwargs.xi.iter();
            let yi: Vec<Option<f64>> = match kwargs.fill_value {
                Some(fill_val) => xi_iter
                    .map(|&xi_val| spline.sample(xi_val).or(Some(fill_val)))
                    .collect(),
                None => xi_iter.map(|&xi_val| spline.sample(xi_val)).collect(),
            };
            Ok(Series::new(fields[1].name(), yi))
        }
        _ => Err(PolarsError::InvalidOperation(
            "`spline` only works on a struct (e.g. pl.struct(col(\"x\"), col(\"y\")))".into(),
        )),
    }
}

fn get_method(method: Option<&str>) -> Result<Interpolation<f64, f64>, PolarsError> {
    match method {
        Some(method_name) => interpolation_from_string(method_name),
        None => Err(PolarsError::InvalidOperation(
            "Please supply a valid `method` kwarg ('linear', 'cosine', or 'catmullrom').".into(),
        )),
    }
}

fn interpolation_from_string(name: &str) -> Result<Interpolation<f64, f64>, PolarsError> {
    match name {
        "linear" => Ok(Interpolation::Linear),
        "cosine" => Ok(Interpolation::Cosine),
        "catmullrom" => Ok(Interpolation::CatmullRom),
        _ => Err(PolarsError::InvalidOperation(
            format!(
                "Invalid method: {}. Valid methods are: 'linear', 'cosine', and 'catmullrom'.",
                name
            )
            .into(),
        )),
    }
}
