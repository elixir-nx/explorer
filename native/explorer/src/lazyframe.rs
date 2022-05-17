use polars::prelude::*;
use std::result::Result;

use crate::{ExDataFrame, ExLazyFrame, ExSeries, ExplorerError};

#[rustler::nif(schedule = "DirtyCpu")]
pub fn lf_collect(data: ExLazyFrame) -> Result<ExDataFrame, ExplorerError> {
    Ok(ExDataFrame::new(data.resource.0.clone().collect()?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn lf_fetch(data: ExLazyFrame, n_rows: usize) -> Result<ExDataFrame, ExplorerError> {
    Ok(ExDataFrame::new(data.resource.0.clone().fetch(n_rows)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn lf_describe_plan(data: ExLazyFrame, optimized: bool) -> Result<String, ExplorerError> {
    let lf = &data.resource.0;
    let plan = match optimized {
        true => lf.describe_optimized_plan()?,
        false => lf.describe_plan(),
    };
    Ok(plan)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn lf_head(data: ExLazyFrame, length: u32) -> Result<ExLazyFrame, ExplorerError> {
    let lf = &data.resource.0;
    Ok(ExLazyFrame::new(lf.clone().limit(length)))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn lf_tail(data: ExLazyFrame, length: u32) -> Result<ExLazyFrame, ExplorerError> {
    let lf = &data.resource.0;
    Ok(ExLazyFrame::new(lf.clone().tail(length)))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn lf_pull(data: ExLazyFrame, name: &str) -> Result<ExSeries, ExplorerError> {
    let df = &data.resource.0.clone().select(&[col(name)]).collect()?;
    let series = df.column(name).map(|s| ExSeries::new(s.clone()))?;
    Ok(series)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn lf_names(data: ExLazyFrame) -> Result<Vec<String>, ExplorerError> {
    let lf = &data.resource.0;
    Ok(lf.schema().iter_names().cloned().collect())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn lf_dtypes(data: ExLazyFrame) -> Result<Vec<String>, ExplorerError> {
    let lf = &data.resource.0;
    Ok(lf
        .schema()
        .iter_dtypes()
        .map(|dtype| dtype.to_string())
        .collect())
}