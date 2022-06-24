defmodule Explorer.PolarsBackend.Native do
  @moduledoc false

  mix_config = Mix.Project.config()
  version = mix_config[:version]
  github_url = mix_config[:package][:links]["GitHub"]

  use RustlerPrecompiled,
    otp_app: :explorer,
    version: version,
    base_url: "#{github_url}/releases/download/v#{version}",
    force_build: System.get_env("EXPLORER_BUILD") in ["1", "true"]

  defstruct [:inner]

  def df_read_csv(
        _filename,
        _infer_schema_length,
        _has_header,
        _stop_after_n_rows,
        _skip_rows,
        _projection,
        _sep,
        _rechunk,
        _columns,
        _dtypes,
        _encoding,
        _null_char,
        _parse_dates
      ),
      do: err()

  def df_to_csv(
        _df,
        _has_headers,
        _delimiter
      ),
      do: err()

  def df_to_csv_file(
        _df,
        _filename,
        _has_headers,
        _delimiter
      ),
      do: err()

  def df_read_ndjson(
        _filename,
        _infer_schema_length,
        _batch_size
      ),
      do: err()

  def df_write_ndjson(
        _df,
        _filename
      ),
      do: err()

  def df_column(_df, _name), do: err()
  def df_columns(_def), do: err()
  def df_drop(_df, _name), do: err()
  def df_drop_duplicates(_df, _maintain_order, _subset), do: err()
  def df_drop_nulls(_df, _subset), do: err()
  def df_dtypes(_df), do: err()
  def df_filter(_df, _mask), do: err()
  def df_get_columns(_df), do: err()
  def df_group_indices(_df, _column_names), do: err()
  def df_groupby_agg(_df, _groups, _aggs), do: err()
  def df_groups(_df, _column_names), do: err()
  def df_head(_df, _length), do: err()
  def df_height(_df), do: err()
  def df_hstack_many(_df, _others), do: err()
  def df_join(_df, _other, _left_on, _right_on, _how, _suffix), do: err()
  def df_melt(_df, _id_vars, _value_vars, _names_to, _values_to), do: err()
  def df_new(_columns), do: err()
  def df_pivot_wider(_df, _id_columns, _pivot_column, _values_column), do: err()
  def df_read_ipc(_filename, _columns, _projection), do: err()
  def df_read_ipc_streaming(_filename, _columns, _projection), do: err()
  def df_read_parquet(_filename), do: err()
  def df_select(_df, _selection), do: err()
  def df_select_at_idx(_df, _idx), do: err()
  def df_set_column_names(_df, _names), do: err()
  def df_shape(_df), do: err()
  def df_slice(_df, _offset, _length), do: err()
  def df_sort(_df, _by, _reverse, _groups), do: err()
  def df_tail(_df, _length), do: err()
  def df_take(_df, _indices), do: err()
  def df_to_dummies(_df, _columns), do: err()
  def df_to_lazy(_df), do: err()
  def df_vstack_many(_df, _others), do: err()
  def df_width(_df), do: err()
  def df_with_columns(_df, _columns), do: err()
  def df_write_ipc(_df, _filename, _compression), do: err()
  def df_write_ipc_streaming(_df, _filename, _compression), do: err()
  def df_write_parquet(_df, _filename), do: err()

  # LazyFrame
  def lf_collect(_df), do: err()
  def lf_describe_plan(_df, _optimized), do: err()
  def lf_drop(_df, _columns), do: err()
  def lf_dtypes(_df), do: err()
  def lf_fetch(_df, _n_rows), do: err()
  def lf_head(_df, _n_rows), do: err()
  def lf_names(_df), do: err()
  def lf_select(_df, _columns), do: err()
  def lf_tail(_df, _n_rows), do: err()

  # Series
  def s_add(_s, _other), do: err()
  def s_and(_s, _s2), do: err()
  def s_append(_s, _other), do: err()
  def s_argsort(_s, _reverse), do: err()
  def s_as_str(_s), do: err()
  def s_cast(_s, _dtype), do: err()
  def s_coalesce(_s, _other), do: err()
  def s_cum_max(_s, _reverse), do: err()
  def s_cum_min(_s, _reverse), do: err()
  def s_cum_sum(_s, _reverse), do: err()
  def s_distinct(_s), do: err()
  def s_div(_s, _other), do: err()
  def s_dtype(_s), do: err()
  def s_eq(_s, _rhs), do: err()
  def s_fill_none(_s, _strategy), do: err()
  def s_fill_none_with_int(_s, _strategy), do: err()
  def s_fill_none_with_float(_s, _strategy), do: err()
  def s_fill_none_with_bin(_s, _strategy), do: err()
  def s_filter(_s, _filter), do: err()
  def s_get(_s, _idx), do: err()
  def s_gt(_s, _rhs), do: err()
  def s_gt_eq(_s, _rhs), do: err()
  def s_head(_s, _length), do: err()
  def s_is_not_null(_s), do: err()
  def s_is_null(_s), do: err()
  def s_len(_s), do: err()
  def s_lt(_s, _rhs), do: err()
  def s_lt_eq(_s, _rhs), do: err()
  def s_max(_s), do: err()
  def s_mean(_s), do: err()
  def s_median(_s), do: err()
  def s_min(_s), do: err()
  def s_mul(_s, _other), do: err()
  def s_n_chunks(_s), do: err()
  def s_name(_s), do: err()
  def s_neq(_s, _rhs), do: err()
  def s_new_bool(_name, _val), do: err()
  def s_new_date32(_name, _val), do: err()
  def s_new_date64(_name, _val), do: err()
  def s_new_f64(_name, _val), do: err()
  def s_new_i64(_name, _val), do: err()
  def s_new_str(_name, _val), do: err()
  def s_or(_s, _s2), do: err()
  def s_peak_max(_s), do: err()
  def s_peak_min(_s), do: err()
  def s_pow(_s, _exponent), do: err()
  def s_int_pow(_s, _exponent), do: err()
  def s_quantile(_s, _quantile, _strategy), do: err()
  def s_rename(_s, _name), do: err()
  def s_reverse(_s), do: err()
  def s_rolling_max(_s, _window_size, _weight, _ignore_null, _min_periods), do: err()
  def s_rolling_mean(_s, _window_size, _weight, _ignore_null, _min_periods), do: err()
  def s_rolling_min(_s, _window_size, _weight, _ignore_null, _min_periods), do: err()
  def s_rolling_sum(_s, _window_size, _weight, _ignore_null, _min_periods), do: err()
  def s_seedable_random_indices(_length, _n_samples, _replacement, _seed), do: err()
  def s_series_equal(_s, _other, _null_equal), do: err()
  def s_slice(_s, _offset, _length), do: err()
  def s_sort(_s, _reverse), do: err()
  def s_std(_s), do: err()
  def s_sub(_s, _other), do: err()
  def s_sum(_s), do: err()
  def s_tail(_s, _length), do: err()
  def s_take(_s, _indices), do: err()
  def s_take_every(_s, _n), do: err()
  def s_to_list(_s), do: err()
  def s_unordered_distinct(_s), do: err()
  def s_value_counts(_s), do: err()
  def s_var(_s), do: err()
  def s_n_unique(_s), do: err()

  defp err, do: :erlang.nif_error(:nif_not_loaded)
end
