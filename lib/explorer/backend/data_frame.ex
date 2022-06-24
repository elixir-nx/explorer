defmodule Explorer.Backend.DataFrame do
  @moduledoc """
  The behaviour for DataFrame backends.
  """

  @type t :: struct()

  @type df :: Explorer.DataFrame.t()
  @type result(t) :: {:ok, t} | {:error, term()}
  @type series :: Explorer.Series.t()
  @type column_name :: String.t()
  @type dtype :: Explorer.Series.dtype()

  @typep basic_types :: float() | integer() | String.t() | Date.t() | DateTime.t()
  @type mutate_value ::
          series()
          | basic_types()
          | [basic_types()]
          | (df() -> series() | basic_types() | [basic_types()])

  # IO

  @callback from_csv(
              filename :: String.t(),
              dtypes :: list({column_name(), dtype()}),
              delimiter :: String.t(),
              null_character :: String.t(),
              skip_rows :: integer(),
              header? :: boolean(),
              encoding :: String.t(),
              max_rows :: integer() | nil,
              columns :: list(column_name()) | list(atom()) | list(integer()) | nil,
              infer_schema_length :: integer() | nil,
              parse_dates :: boolean()
            ) :: result(df)
  @callback to_csv(df, filename :: String.t(), header? :: boolean(), delimiter :: String.t()) ::
              result(String.t())

  @callback from_parquet(filename :: String.t()) :: result(df)
  @callback to_parquet(df, filename :: String.t()) :: result(String.t())

  @callback from_ipc(
              filename :: String.t(),
              columns :: list(String.t()) | list(atom()) | list(integer()) | nil
            ) :: result(df)
  @callback to_ipc(df, filename :: String.t(), compression :: String.t()) ::
              result(String.t())

  @callback from_ipc_streaming(
              filename :: String.t(),
              columns :: list(String.t()) | list(atom()) | list(integer()) | nil
            ) :: result(df)
  @callback to_ipc_streaming(df, filename :: String.t(), compression :: String.t()) ::
              result(String.t())

  @callback from_ndjson(
              filename :: String.t(),
              infer_schema_length :: integer(),
              batch_size :: integer()
            ) :: result(df)
  @callback to_ndjson(df, filename :: String.t()) :: result(String.t())

  # Conversion

  @callback lazy() :: module()
  @callback to_lazy(df) :: df
  @callback collect(df) :: df
  @callback from_tabular(Table.Reader.t()) :: df
  @callback from_series(map() | Keyword.t()) :: df
  @callback to_rows(df, atom_keys? :: boolean()) :: [map()]
  @callback dump_csv(df, header? :: boolean(), delimiter :: String.t()) :: String.t()

  # Introspection

  @callback n_rows(df) :: integer()
  @callback inspect(df, opts :: Inspect.Opts.t()) :: Inspect.Algebra.t()

  # Single table verbs

  @callback head(df, rows :: integer()) :: df
  @callback tail(df, rows :: integer()) :: df
  @callback select(df, out_df :: df()) :: df
  @callback filter(df, mask :: series) :: df
  @callback mutate(df, out_df :: df(), mutations :: [{column_name(), mutate_value()}]) :: df
  @callback arrange(df, columns :: [column_name() | {:asc | :desc, column_name()}]) :: df
  @callback distinct(df, out_df :: df(), columns :: [column_name()], keep_all? :: boolean()) :: df
  @callback rename(df, out_df :: df()) :: df
  @callback dummies(df, columns :: [column_name()]) :: df
  @callback sample(df, n :: integer(), replacement :: boolean(), seed :: integer()) :: df
  @callback pull(df, column :: column_name()) :: series
  @callback slice(df, offset :: integer(), length :: integer()) :: df
  @callback take(df, indices :: list(integer())) :: df
  @callback drop_nil(df, columns :: [column_name()]) :: df
  @callback pivot_wider(
              df,
              id_columns :: [column_name()],
              names_from :: column_name(),
              values_from :: column_name(),
              names_prefix :: String.t()
            ) :: df
  @callback pivot_longer(
              df,
              out_df :: df(),
              columns_to_pivot :: [column_name()],
              columns_to_keep :: [column_name()],
              names_to :: column_name(),
              values_to :: column_name()
            ) :: df

  # Two or more table verbs

  @callback join(
              left :: df(),
              right :: df(),
              out_df :: df(),
              on :: list({column_name(), column_name()}),
              how :: :left | :inner | :outer | :right | :cross
            ) :: df

  @callback concat_columns([df]) :: df
  @callback concat_rows([df]) :: df

  # Groups

  @callback summarise(df, out_df :: df(), aggregations :: %{column_name() => [atom()]}) :: df

  # Functions
  alias Explorer.{DataFrame, Series}

  @doc """
  Creates a new DataFrame for a given backend.
  """
  def new(data, names, dtypes) do
    dtypes_pairs = Enum.zip(names, dtypes)
    %DataFrame{data: data, names: names, dtypes: Map.new(dtypes_pairs), groups: []}
  end

  @default_limit 5
  import Inspect.Algebra

  @doc """
  Default inspect implementation for backends.
  """
  def inspect(df, backend, n_rows, inspect_opts, opts \\ [])
      when is_binary(backend) and (is_integer(n_rows) or is_nil(n_rows)) and is_list(opts) do
    inspect_opts = %{inspect_opts | limit: @default_limit}
    open = color("[", :list, inspect_opts)
    close = color("]", :list, inspect_opts)

    cols_algebra =
      for name <- DataFrame.names(df) do
        series = df[name]

        values =
          series
          |> Series.slice(0, inspect_opts.limit + 1)
          |> Series.to_list()

        data = container_doc(open, values, close, inspect_opts, &Explorer.Shared.to_string/2)

        concat([
          line(),
          color("#{name} ", :map, inspect_opts),
          color("#{Series.dtype(series)}", :atom, inspect_opts),
          " ",
          data
        ])
      end

    concat([
      color(backend, :atom, inspect_opts),
      open,
      "#{n_rows || "???"} x #{length(cols_algebra)}",
      close,
      groups_algebra(df.groups, inspect_opts) | cols_algebra
    ])
  end

  defp groups_algebra([_ | _] = groups, opts),
    do:
      Inspect.Algebra.concat([
        Inspect.Algebra.line(),
        Inspect.Algebra.color("Groups: ", :atom, opts),
        Inspect.Algebra.to_doc(groups, opts)
      ])

  defp groups_algebra([], _), do: ""
end
