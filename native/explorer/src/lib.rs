// MiMalloc won´t compile on Windows with the GCC compiler.
// On Linux with Musl it won´t load correctly.
#[cfg(not(any(
    all(windows, target_env = "gnu"),
    all(target_os = "linux", target_env = "musl")
)))]
use mimalloc::MiMalloc;
use rustler::{Env, Term};

#[cfg(not(any(
    all(windows, target_env = "gnu"),
    all(target_os = "linux", target_env = "musl")
)))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod dataframe;
#[allow(clippy::extra_unused_lifetimes)]
mod datatypes;
mod error;
mod lazyframe;
mod series;

use dataframe::*;
pub use datatypes::{
    ExDataFrame, ExDataFrameRef, ExLazyFrame, ExLazyFrameRef, ExSeries, ExSeriesRef,
};
pub use error::ExplorerError;
use lazyframe::*;
use series::*;

fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(ExDataFrameRef, env);
    rustler::resource!(ExLazyFrameRef, env);
    rustler::resource!(ExSeriesRef, env);
    true
}

mod atoms {
    rustler::atoms! {
        calendar = "Elixir.Calendar.ISO"
    }
}

rustler::init!(
    "Elixir.Explorer.PolarsBackend.Native",
    [
        df_column,
        df_columns,
        df_drop,
        df_drop_duplicates,
        df_drop_nulls,
        df_dtypes,
        df_filter,
        df_get_columns,
        df_groups,
        df_groupby_agg,
        df_group_indices,
        df_head,
        df_height,
        df_hstack_many,
        df_join,
        df_melt,
        df_new,
        df_pivot_wider,
        df_read_csv,
        df_read_ipc,
        df_read_ipc_stream,
        df_read_parquet,
        df_read_ndjson,
        df_write_ndjson,
        df_select,
        df_select_at_idx,
        df_set_column_names,
        df_shape,
        df_slice,
        df_sort,
        df_tail,
        df_take,
        df_to_csv,
        df_to_csv_file,
        df_to_dummies,
        df_to_lazy,
        df_vstack_many,
        df_width,
        df_with_columns,
        df_write_ipc,
        df_write_ipc_stream,
        df_write_parquet,
        // lazyframe
        lf_collect,
        lf_describe_plan,
        lf_drop,
        lf_dtypes,
        lf_fetch,
        lf_head,
        lf_names,
        lf_select,
        lf_tail,
        // series
        s_add,
        s_and,
        s_append,
        s_argsort,
        s_as_str,
        s_cast,
        s_coalesce,
        s_cum_max,
        s_cum_min,
        s_cum_sum,
        s_distinct,
        s_div,
        s_dtype,
        s_eq,
        s_fill_none,
        s_fill_none_with_int,
        s_fill_none_with_float,
        s_fill_none_with_bin,
        s_filter,
        s_get,
        s_gt,
        s_gt_eq,
        s_head,
        s_is_not_null,
        s_is_null,
        s_len,
        s_lt,
        s_lt_eq,
        s_max,
        s_mean,
        s_median,
        s_min,
        s_mul,
        s_n_unique,
        s_name,
        s_neq,
        s_new_bool,
        s_new_date32,
        s_new_date64,
        s_new_f64,
        s_new_i64,
        s_new_str,
        s_or,
        s_peak_max,
        s_peak_min,
        s_pow,
        s_int_pow,
        s_quantile,
        s_rename,
        s_reverse,
        s_rolling_max,
        s_rolling_mean,
        s_rolling_min,
        s_rolling_sum,
        s_seedable_random_indices,
        s_series_equal,
        s_slice,
        s_sort,
        s_std,
        s_sub,
        s_sum,
        s_tail,
        s_take,
        s_take_every,
        s_to_list,
        s_unordered_distinct,
        s_var,
        s_value_counts,
    ],
    load = on_load
);
