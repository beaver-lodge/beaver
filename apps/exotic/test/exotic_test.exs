defmodule ExoticTest.LibC.TM do
  use Exotic.Type.Struct,
    fields: [
      tm_sec: :i32,
      tm_min: :i32,
      tm_hour: :i32,
      tm_mday: :i32,
      tm_mon: :i32,
      tm_year: :i32,
      tm_wday: :i32,
      tm_yday: :i32,
      tm_isdst: :i32,
      __tm_gmtoff__: :i64,
      __tm_zone__: :ptr
    ]
end

defmodule ExoticTest do
  use ExUnit.Case
  doctest Exotic
  alias Exotic.LibC

  setup_all do
    [libc: LibC.load!()]
  end

  test "test arith" do
    assert LibC.sin(3.1415926535 * 0.5) |> Exotic.Value.extract() == 1.0
    assert LibC.cos(3.1415926535) |> Exotic.Value.extract() == -1.0
  end

  test "test extract" do
    assert Exotic.Value.get(3.1415926535) |> Exotic.Value.extract() == 3.1415926535
  end

  test "use libc to print a elixir string" do
    array =
      [1, 2, 3]
      |> Exotic.Value.Array.get()

    for _ <- 0..200 do
      <<1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0>> =
        array
        |> Exotic.Value.as_binary()
    end
  end

  test "extract struct" do
    v =
      [1, 2, 3]
      |> Exotic.Value.Array.get()

    struct_type =
      [Exotic.NIF.get_u32_type(), Exotic.NIF.get_u32_type(), Exotic.NIF.get_u32_type()]
      |> Exotic.NIF.get_struct_type()

    assert [1, 2, 3] ==
             Exotic.Value.Struct.extract(struct_type, v)
  end

  test "extract opaque" do
    assert [1, 2, 3, 4, 5, 6, 7, 8]
           |> Exotic.Value.Array.get()
           |> Exotic.Value.as_binary() == <<
             1,
             0,
             0,
             0,
             2,
             0,
             0,
             0,
             3,
             0,
             0,
             0,
             4,
             0,
             0,
             0,
             5,
             0,
             0,
             0,
             6,
             0,
             0,
             0,
             7,
             0,
             0,
             0,
             8,
             0,
             0,
             0
           >>
  end
end
