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
fn interpolate(inputs: &[Series], kwargs: SplineKwargs) -> PolarsResult<Series> {
    if inputs.len() != 1 {
        return Err(PolarsError::InvalidOperation(
            "`spline` only works on a single struct".into(),
        ));
    }
    let input = &inputs[0];
    match input.dtype() {
        DataType::Struct(_) => {
            let spline = create_spline(input, kwargs.method.as_deref())?;

            let yi: Vec<Option<f64>> = sample_spline(&spline, &kwargs)?;

            let fields = input.struct_()?.fields();
            Ok(Series::new(fields[1].name(), yi))
        }
        _ => Err(PolarsError::InvalidOperation(
            "`spline` only works on a struct (e.g. pl.struct(col(\"x\"), col(\"y\")))".into(),
        )),
    }
}

fn create_spline(input: &Series, method: Option<&str>) -> PolarsResult<Spline<f64, f64>> {
    let fields = input.struct_()?.fields();
    let x: Vec<f64> = fields[0].f64()?.into_iter().flatten().collect();
    let y: Vec<f64> = fields[1].f64()?.into_iter().flatten().collect();

    let interpolator = interpolator_from_name(method)?;

    let keys: Vec<Key<f64, f64>> = x
        .into_iter()
        .zip(y.into_iter())
        .map(|(x, y)| Key::new(x, y, interpolator))
        .collect();
    Ok(Spline::from_vec(keys))
}

fn sample_spline(
    spline: &Spline<f64, f64>,
    kwargs: &SplineKwargs,
) -> PolarsResult<Vec<Option<f64>>> {
    let xi_iter = kwargs.xi.iter();
    let yi: Vec<Option<f64>> = match kwargs.fill_value {
        Some(fill_val) => xi_iter
            .map(|&xi_val| spline.sample(xi_val).or(Some(fill_val)))
            .collect(),
        None => xi_iter.map(|&xi_val| spline.sample(xi_val)).collect(),
    };
    Ok(yi)
}

fn interpolator_from_name(method: Option<&str>) -> Result<Interpolation<f64, f64>, PolarsError> {
    match method {
        Some("linear") => Ok(Interpolation::Linear),
        Some("cosine") => Ok(Interpolation::Cosine),
        Some("catmullrom") => Ok(Interpolation::CatmullRom),
        Some(invalid) => Err(PolarsError::InvalidOperation(
            format!(
                "Invalid method: {}. Valid methods are: 'linear', 'cosine', and 'catmullrom'.",
                invalid
            )
            .into(),
        )),
        None => Err(PolarsError::InvalidOperation(
            "Please supply a valid `method` kwarg ('linear', 'cosine', or 'catmullrom').".into(),
        )),
    }
}
