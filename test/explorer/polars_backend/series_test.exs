defmodule Explorer.PolarsBackend.SeriesTest do
  use ExUnit.Case, async: true
  alias Explorer.PolarsBackend.Series

  test "from_list/2 of dates" do
    dates = [~D[1643-01-04], ~D[-0030-08-12], ~D[1994-05-01]]

    assert Series.from_list(dates, :date) |> Series.to_list() == dates

    today_in_days = Date.utc_today() |> Date.to_gregorian_days()

    dates =
      for _i <- 0..:rand.uniform(100) do
        days = :rand.uniform(today_in_days)

        Date.from_gregorian_days(days)
      end

    assert Series.from_list(dates, :date) |> Series.to_list() == dates

    dates = Enum.intersperse(dates, nil)
    assert Series.from_list(dates, :date) |> Series.to_list() == dates
  end

  test "from_list/2 of naive datetime" do
    dates = [
      ~N[2022-04-13 15:44:31.560227],
      ~N[1022-01-04 21:18:31.224123],
      ~N[1988-11-23 06:36:16.158432],
      ~N[2353-03-07 00:39:35.702789]
    ]

    assert Series.from_list(dates, :datetime) |> Series.to_list() == dates

    today_in_days = Date.utc_today() |> Date.to_gregorian_days()
    day_in_seconds = 86_400

    dates =
      for _i <- 0..:rand.uniform(100) do
        days = :rand.uniform(today_in_days)
        seconds = days * day_in_seconds
        microseconds = {:rand.uniform(999_999), 6}

        seconds
        |> NaiveDateTime.from_gregorian_seconds(microseconds)
        |> NaiveDateTime.add(:rand.uniform(24) * 60 * 60, :second)
        |> NaiveDateTime.add(:rand.uniform(60) * 60, :second)
        |> NaiveDateTime.add(:rand.uniform(60), :second)
      end

    assert Series.from_list(dates, :datetime) |> Series.to_list() == dates
  end

  test "to_enum/1 returns a valid enumerable" do
    enum1 =
      [1, 2, 3, 4]
      |> Explorer.Series.from_list(backend: Explorer.PolarsBackend)
      |> Explorer.Series.to_enum()

    enum2 =
      ["a", "b", "c"]
      |> Explorer.Series.from_list(backend: Explorer.PolarsBackend)
      |> Explorer.Series.to_enum()

    assert Enum.zip(enum1, enum2) == [{1, "a"}, {2, "b"}, {3, "c"}]

    assert Enum.reduce(enum1, 0, &+/2) == 10
    assert Enum.reduce(enum2, "", &<>/2) == "cba"

    assert Enum.count(enum1) == 4
    assert Enum.count(enum2) == 3

    assert Enum.slice(enum1, 1..2) == [2, 3]
    assert Enum.slice(enum2, 1..2) == ["b", "c"]
  end
end
