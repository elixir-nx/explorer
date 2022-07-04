defmodule Explorer.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/explorer"
  @version "0.3.0-dev"

  def project do
    [
      app: :explorer,
      name: "Explorer",
      description:
        "Series (one-dimensional) and dataframes (two-dimensional) for fast data exploration in Elixir",
      version: @version,
      elixir: "~> 1.13",
      package: package(),
      deps: deps(),
      docs: docs(),
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      env: [default_backend: Explorer.PolarsBackend]
    ]
  end

  defp deps do
    [
      {:ex_doc, "~> 0.24", only: :docs, runtime: false},
      {:nx, "~> 0.1.0", only: :test},
      {:rustler_precompiled, "~> 0.4"},
      {:rustler, ">= 0.0.0", optional: true},
      {:table, "~> 0.1.2"},
      {:table_rex, "~> 3.1.1"}
    ]
  end

  defp docs do
    [
      main: "Explorer",
      logo: "explorer-exdoc.png",
      source_ref: "v#{@version}",
      source_url: @source_url,
      groups_for_modules: [
        # Explorer,
        # Explorer.DataFrame,
        # Explorer.Datasets,
        # Explorer.Series,
        Backends: [
          Explorer.Backend,
          Explorer.Backend.DataFrame,
          Explorer.Backend.Series,
          Explorer.PolarsBackend
        ]
      ],
      groups_for_functions: [
        "Functions: Single-table": &(&1[:type] == :single),
        "Functions: Multi-table": &(&1[:type] == :multi),
        "Functions: Introspection": &(&1[:type] == :introspection),
        "Functions: IO": &(&1[:type] == :io)
      ],
      extras: ["notebooks/exploring_explorer.livemd", "CHANGELOG.md"],
      skip_undefined_reference_warnings_on: ["CHANGELOG.md"]
    ]
  end

  defp package do
    [
      files: [
        "lib",
        "native",
        "datasets",
        "checksum-*.exs",
        "mix.exs",
        "LICENSE"
      ],
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      maintainers: ["Christopher Grainger"]
    ]
  end
end
