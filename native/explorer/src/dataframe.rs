use polars::prelude::*;

use rustler::{Binary, Env, NewBinary};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::result::Result;

use crate::series::{to_ex_series_collection, to_series_collection};

use crate::{ExDataFrame, ExLazyFrame, ExSeries, ExplorerError};

#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
pub fn df_read_csv(
    filename: &str,
    infer_schema_length: Option<usize>,
    has_header: bool,
    stop_after_n_rows: Option<usize>,
    skip_rows: usize,
    projection: Option<Vec<usize>>,
    sep: &str,
    do_rechunk: bool,
    column_names: Option<Vec<String>>,
    dtypes: Option<Vec<(&str, &str)>>,
    encoding: &str,
    null_char: String,
    parse_dates: bool,
) -> Result<ExDataFrame, ExplorerError> {
    let encoding = match encoding {
        "utf8-lossy" => CsvEncoding::LossyUtf8,
        _ => CsvEncoding::Utf8,
    };

    let schema: Option<Schema> = match dtypes {
        Some(dtypes) => {
            let mut schema = Schema::new();
            for (name, dtype) in dtypes {
                schema.with_column(name.to_string(), dtype_from_str(dtype)?)
            }
            Some(schema)
        }
        None => None,
    };

    let df = CsvReader::from_path(filename)?
        .infer_schema(infer_schema_length)
        .has_header(has_header)
        .with_parse_dates(parse_dates)
        .with_n_rows(stop_after_n_rows)
        .with_delimiter(sep.as_bytes()[0])
        .with_skip_rows(skip_rows)
        .with_projection(projection)
        .with_rechunk(do_rechunk)
        .with_encoding(encoding)
        .with_columns(column_names)
        .with_dtypes(schema.as_ref())
        .with_null_values(Some(NullValues::AllColumns(null_char)))
        .finish()?;

    Ok(ExDataFrame::new(df))
}

fn dtype_from_str(dtype: &str) -> Result<DataType, ExplorerError> {
    match dtype {
        "str" => Ok(DataType::Utf8),
        "f64" => Ok(DataType::Float64),
        "i64" => Ok(DataType::Int64),
        "bool" => Ok(DataType::Boolean),
        "date" => Ok(DataType::Date),
        "datetime[μs]" => Ok(DataType::Datetime(TimeUnit::Microseconds, None)),
        "datetime[ms]" => Ok(DataType::Datetime(TimeUnit::Milliseconds, None)),
        _ => Err(ExplorerError::Internal("Unrecognised datatype".into())),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
pub fn df_read_parquet(filename: &str) -> Result<ExDataFrame, ExplorerError> {
    let file = File::open(filename)?;
    let buf_reader = BufReader::new(file);
    let df = ParquetReader::new(buf_reader).finish()?;
    Ok(ExDataFrame::new(df))
}

#[rustler::nif(schedule = "DirtyIo")]
pub fn df_write_parquet(data: ExDataFrame, filename: &str) -> Result<(), ExplorerError> {
    let df = &data.resource.0;
    let file = File::create(filename)?;
    let mut buf_writer = BufWriter::new(file);
    ParquetWriter::new(&mut buf_writer).finish(&mut df.clone())?;
    Ok(())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_to_csv(
    env: Env,
    data: ExDataFrame,
    has_headers: bool,
    delimiter: u8,
) -> Result<Binary, ExplorerError> {
    let df = &data.resource.0;
    let mut buf = vec![];

    CsvWriter::new(&mut buf)
        .has_header(has_headers)
        .with_delimiter(delimiter)
        .finish(&mut df.clone())?;

    let mut values_binary = NewBinary::new(env, buf.len());
    values_binary.copy_from_slice(&buf);

    Ok(values_binary.into())
}

#[rustler::nif(schedule = "DirtyIo")]
pub fn df_to_csv_file(
    data: ExDataFrame,
    filename: &str,
    has_headers: bool,
    delimiter: u8,
) -> Result<(), ExplorerError> {
    let df = &data.resource.0;
    let file = File::create(filename)?;
    let mut buf_writer = BufWriter::new(file);
    CsvWriter::new(&mut buf_writer)
        .has_header(has_headers)
        .with_delimiter(delimiter)
        .finish(&mut df.clone())?;
    Ok(())
}

#[rustler::nif(schedule = "DirtyIo")]
pub fn df_read_ipc(
    filename: &str,
    columns: Option<Vec<String>>,
    projection: Option<Vec<usize>>,
) -> Result<ExDataFrame, ExplorerError> {
    let file = File::open(filename)?;
    let buf_reader = BufReader::new(file);
    let df = IpcReader::new(buf_reader)
        .with_columns(columns)
        .with_projection(projection)
        .finish()?;
    Ok(ExDataFrame::new(df))
}

#[rustler::nif(schedule = "DirtyIo")]
pub fn df_write_ipc(
    data: ExDataFrame,
    filename: &str,
    compression: Option<&str>,
) -> Result<(), ExplorerError> {
    let df = &data.resource.0;
    // Select the compression algorithm.
    let compression = match compression {
        Some("LZ4") => Some(IpcCompression::LZ4),
        Some("ZSTD") => Some(IpcCompression::ZSTD),
        _ => None,
    };

    let file = File::create(filename)?;
    let mut buf_writer = BufWriter::new(file);
    IpcWriter::new(&mut buf_writer)
        .with_compression(compression)
        .finish(&mut df.clone())?;
    Ok(())
}

#[rustler::nif(schedule = "DirtyIo")]
pub fn df_read_ipc_stream(
    filename: &str,
    columns: Option<Vec<String>>,
    projection: Option<Vec<usize>>,
) -> Result<ExDataFrame, ExplorerError> {
    let f = File::open(filename)?;
    let df = IpcStreamReader::new(f)
        .with_columns(columns)
        .with_projection(projection)
        .finish()?;
    Ok(ExDataFrame::new(df))
}

#[rustler::nif(schedule = "DirtyIo")]
pub fn df_write_ipc_stream(
    data: ExDataFrame,
    filename: &str,
    compression: Option<&str>,
) -> Result<(), ExplorerError> {
    let df = &data.resource.0;
    // Select the compression algorithm.
    let compression = match compression {
        Some("LZ4") => Some(IpcCompression::LZ4),
        Some("ZSTD") => Some(IpcCompression::ZSTD),
        _ => None,
    };

    let mut file = File::create(filename).expect("could not create file");
    IpcStreamWriter::new(&mut file)
        .with_compression(compression)
        .finish(&mut df.clone())?;
    Ok(())
}

#[rustler::nif(schedule = "DirtyIo")]
pub fn df_read_ndjson(
    filename: &str,
    infer_schema_length: Option<usize>,
    batch_size: usize,
) -> Result<ExDataFrame, ExplorerError> {
    let file = File::open(filename)?;
    let buf_reader = BufReader::new(file);
    let df = JsonReader::new(buf_reader)
        .with_json_format(JsonFormat::JsonLines)
        .with_batch_size(batch_size)
        .infer_schema_len(infer_schema_length)
        .finish()?;

    Ok(ExDataFrame::new(df))
}

#[rustler::nif(schedule = "DirtyIo")]
pub fn df_write_ndjson(data: ExDataFrame, filename: &str) -> Result<(), ExplorerError> {
    let df = &data.resource.0;
    let file = File::create(filename)?;
    let mut buf_writer = BufWriter::new(file);
    JsonWriter::new(&mut buf_writer).finish(&mut df.clone())?;
    Ok(())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_join(
    data: ExDataFrame,
    other: ExDataFrame,
    left_on: Vec<&str>,
    right_on: Vec<&str>,
    how: &str,
    suffix: Option<String>,
) -> Result<ExDataFrame, ExplorerError> {
    let how = match how {
        "left" => JoinType::Left,
        "inner" => JoinType::Inner,
        "outer" => JoinType::Outer,
        "cross" => JoinType::Cross,
        _ => {
            return Err(ExplorerError::Other(format!(
                "Join method {} not supported",
                how
            )))
        }
    };

    let df = &data.resource.0;
    let df1 = &other.resource.0;
    let new_df = df.join(df1, left_on, right_on, how, suffix)?;
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_get_columns(data: ExDataFrame) -> Result<Vec<ExSeries>, ExplorerError> {
    let df = &data.resource.0;
    Ok(to_ex_series_collection(df.get_columns().clone()))
}

#[rustler::nif]
pub fn df_columns(data: ExDataFrame) -> Result<Vec<String>, ExplorerError> {
    let df = &data.resource.0;
    Ok(df.get_column_names_owned())
}

#[rustler::nif]
pub fn df_dtypes(data: ExDataFrame) -> Result<Vec<String>, ExplorerError> {
    let df = &data.resource.0;
    let result = df.dtypes().iter().map(|dtype| dtype.to_string()).collect();
    Ok(result)
}

#[rustler::nif]
pub fn df_shape(data: ExDataFrame) -> Result<(usize, usize), ExplorerError> {
    Ok(data.resource.0.shape())
}

#[rustler::nif]
pub fn df_height(data: ExDataFrame) -> Result<usize, ExplorerError> {
    Ok(data.resource.0.height())
}

#[rustler::nif]
pub fn df_width(data: ExDataFrame) -> Result<usize, ExplorerError> {
    Ok(data.resource.0.width())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_vstack_many(
    data: ExDataFrame,
    others: Vec<ExDataFrame>,
) -> Result<ExDataFrame, ExplorerError> {
    let mut out_df = data.resource.0.clone();
    let names = out_df.get_column_names();
    let dfs = others
        .into_iter()
        .map(|ex_df| ex_df.resource.0.select(&names))
        .collect::<Result<Vec<_>, _>>()?;

    for df in dfs {
        out_df.vstack_mut(&df)?;
    }
    // Follows recommendation from docs and rechunk after many vstacks.
    out_df.rechunk();
    Ok(ExDataFrame::new(out_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_hstack_many(
    data: ExDataFrame,
    others: Vec<ExDataFrame>,
) -> Result<ExDataFrame, ExplorerError> {
    let id_column = "__row_count_id__";
    let first = data
        .resource
        .0
        .clone()
        .lazy()
        .with_row_count(id_column, None);

    // We need to be able to handle arbitrary column name overlap.
    // This builds up a join and suffixes conflicting names with _N where
    // N is the index of the df in the join array.
    let (out_df, _) = others
        .iter()
        .map(|data| {
            data.resource
                .0
                .clone()
                .lazy()
                .with_row_count(id_column, None)
        })
        .fold((first, 1), |(acc_df, count), df| {
            let suffix = format!("_{}", count);
            let new_df = acc_df
                .join_builder()
                .with(df)
                .how(JoinType::Inner)
                .left_on([col(id_column)])
                .right_on([col(id_column)])
                .suffix(suffix)
                .finish();
            (new_df, count + 1)
        });

    Ok(ExDataFrame::new(
        out_df.drop_columns([id_column]).collect()?,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_drop_nulls(
    data: ExDataFrame,
    subset: Option<Vec<String>>,
) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let new_df = df.drop_nulls(subset.as_ref().map(|s| s.as_ref()))?;
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_drop(data: ExDataFrame, name: &str) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let new_df = df.drop(name)?;
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_select_at_idx(data: ExDataFrame, idx: usize) -> Result<Option<ExSeries>, ExplorerError> {
    let df = &data.resource.0;
    let result = df.select_at_idx(idx).map(|s| ExSeries::new(s.clone()));
    Ok(result)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_column(data: ExDataFrame, name: &str) -> Result<ExSeries, ExplorerError> {
    let df = &data.resource.0;
    let series = df.column(name).map(|s| ExSeries::new(s.clone()))?;
    Ok(series)
}

#[rustler::nif]
pub fn df_select(data: ExDataFrame, selection: Vec<&str>) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let new_df = df.select(&selection)?;
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_filter(data: ExDataFrame, mask: ExSeries) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let filter_series = &mask.resource.0;
    if let Ok(ca) = filter_series.bool() {
        let new_df = df.filter(ca)?;
        Ok(ExDataFrame::new(new_df))
    } else {
        Err(ExplorerError::Other("Expected a boolean mask".into()))
    }
}

#[rustler::nif]
pub fn df_take(data: ExDataFrame, indices: Vec<u32>) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let idx = UInt32Chunked::from_vec("idx", indices);
    let new_df = df.take(&idx)?;
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_sort(
    data: ExDataFrame,
    by_columns: Vec<String>,
    reverse: Vec<bool>,
    groups: Vec<String>,
) -> Result<ExDataFrame, ExplorerError> {
    let df: DataFrame = data.resource.0.clone();

    let new_df = if groups.is_empty() {
        let new_df = &df.sort(by_columns, reverse)?;
        new_df.clone()
    } else {
        let new_df = &df
            .groupby(groups)?
            .apply(|df| df.sort(by_columns.clone(), reverse.clone()))?;
        new_df.clone()
    };

    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_slice(
    data: ExDataFrame,
    offset: i64,
    length: usize,
) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let new_df = df.slice(offset, length);
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_head(data: ExDataFrame, length: Option<usize>) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let new_df = df.head(length);
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_tail(data: ExDataFrame, length: Option<usize>) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let new_df = df.tail(length);
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_melt(
    data: ExDataFrame,
    id_vars: Vec<String>,
    value_vars: Vec<String>,
    names_to: String,
    values_to: String,
) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let melt_opts = MeltArgs {
        id_vars,
        value_vars,
        variable_name: Some(names_to),
        value_name: Some(values_to),
    };
    let new_df = df.melt2(melt_opts)?;
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_drop_duplicates(
    data: ExDataFrame,
    maintain_order: bool,
    subset: Vec<String>,
) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let new_df = match maintain_order {
        false => df.unique(Some(&subset), UniqueKeepStrategy::First)?,
        true => df.unique_stable(Some(&subset), UniqueKeepStrategy::First)?,
    };
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_to_dummies(
    data: ExDataFrame,
    selection: Vec<&str>,
) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let new_df = df.select(&selection).and_then(|df| df.to_dummies())?;
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_with_columns(
    data: ExDataFrame,
    columns: Vec<ExSeries>,
) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let mut new_df = df.clone();

    for column in columns {
        let series = column.resource.0.clone();
        new_df = new_df.with_column(series)?.clone();
    }

    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif]
pub fn df_new(columns: Vec<ExSeries>) -> Result<ExDataFrame, ExplorerError> {
    let columns = to_series_collection(columns);
    let df = DataFrame::new(columns)?;
    Ok(ExDataFrame::new(df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_set_column_names(
    data: ExDataFrame,
    names: Vec<&str>,
) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let mut new_df = df.clone();
    new_df.set_column_names(&names)?;
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_groups(data: ExDataFrame, groups: Vec<&str>) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let groups = df.groupby(groups)?.groups()?;
    Ok(ExDataFrame::new(groups))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_group_indices(data: ExDataFrame, groups: Vec<&str>) -> Result<ExSeries, ExplorerError> {
    let df = &data.resource.0;
    let series = df
        .groupby(groups)?
        .groups()?
        .column("groups")
        .map(|series| ExSeries::new(series.clone()))?;
    Ok(series)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_groupby_agg(
    data: ExDataFrame,
    groups: Vec<&str>,
    aggs: Vec<(&str, Vec<&str>)>,
) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let new_df = df.groupby_stable(groups)?.agg(&aggs)?;
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_pivot_wider(
    data: ExDataFrame,
    id_columns: Vec<&str>,
    pivot_column: &str,
    values_column: &str,
) -> Result<ExDataFrame, ExplorerError> {
    let df = &data.resource.0;
    let new_df = df.pivot_stable(
        [values_column],
        id_columns,
        [pivot_column],
        PivotAgg::First,
        false,
    )?;
    Ok(ExDataFrame::new(new_df))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn df_to_lazy(data: ExDataFrame) -> Result<ExLazyFrame, ExplorerError> {
    let new_lf = data.resource.0.clone().lazy();
    Ok(ExLazyFrame::new(new_lf))
}
