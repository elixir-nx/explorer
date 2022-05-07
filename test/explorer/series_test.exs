defmodule Explorer.SeriesTest do
  use ExUnit.Case, async: true

  alias Explorer.Series

  doctest Explorer.Series

  test "from_list/1" do
    s = Series.from_list([1, 2, 3])

    assert Series.to_list(s) === [1, 2, 3]

    s = Series.from_list([1, 2.4, 3])
    assert Series.to_list(s) === [1.0, 2.4, 3.0]

    assert_raise ArgumentError, fn ->
      s = Series.from_list([1, "foo", 3])
      Series.to_list(s)
    end
  end

  test "fetch/2" do
    s = Series.from_list([1, 2, 3])
    assert s[0] === 1
    assert s[0..1] |> Series.to_list() === [1, 2]
    assert s[[0, 1]] |> Series.to_list() === [1, 2]
  end

  test "pop/2" do
    s = Series.from_list([1, 2, 3])
    assert {1, %Series{} = s1} = Access.pop(s, 0)
    assert Series.to_list(s1) == [2, 3]
    assert {%Series{} = s2, %Series{} = s3} = Access.pop(s, 0..1)
    assert Series.to_list(s2) == [1, 2]
    assert Series.to_list(s3) == [3]
    assert {%Series{} = s4, %Series{} = s5} = Access.pop(s, [0, 1])
    assert Series.to_list(s4) == [1, 2]
    assert Series.to_list(s5) == [3]
  end

  test "get_and_update/3" do
    s = Series.from_list([1, 2, 3])

    assert {2, %Series{} = s2} =
             Access.get_and_update(s, 1, fn current_value ->
               {current_value, current_value * 2}
             end)

    assert Series.to_list(s2) == [1, 4, 3]
  end

  describe "equal/2" do
    test "returns an empty series if both are empty" do
      left = Series.from_list([])
      right = Series.from_list([])
      s = Series.equal(left, right)
      assert Series.to_list(s) == []
    end
  end

  test "raises error when RHS is an empty Series" do
    assert_raise ArgumentError,
                 "Explorer.Series.equal/2 cannot compare with zero length series",
                 fn ->
                   left = Series.from_list([1, 2, 3])
                   right = Series.from_list([])
                   Series.equal(left, right)
                 end
  end

  test "raises error when LHS is an empty Series" do
    assert_raise ArgumentError,
                 "Explorer.Series.equal/2 cannot compare with zero length series",
                 fn ->
                   left = Series.from_list([])
                   right = Series.from_list([1, 2, 3])
                   Series.equal(left, right)
                 end
  end

  test "do not raise any error if LHS is length 1" do
    left = Series.from_list([1])
    right = Series.from_list([1, 2, 3])
    s = Series.equal(left, right)
    assert Series.to_list(s) == [true, false, false]
  end
end
