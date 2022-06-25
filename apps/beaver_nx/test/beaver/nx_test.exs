defmodule BeaverNxTest do
  @moduledoc """
  Tests for compliance with the Nx backend behavior. Many of these tests are adapted from EXLA
  """
  use ExUnit.Case, async: true
  doctest Beaver.Nx

  setup do
    Nx.default_backend(Beaver.Nx)
    :ok
  end

  test "Nx.to_binary/1" do
    t = Nx.tensor([1, 2, 3, 4], backend: Beaver.Nx)
    assert Nx.to_binary(t) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
    assert Nx.to_binary(t, limit: 2) == <<1::64-native, 2::64-native>>
    assert Nx.to_binary(t, limit: 6) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
  end

  test "Nx.backend_transfer/1" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, {Beaver.Nx, device_id: 0})
    assert %Beaver.Nx{memref: %Exotic.Value.Struct{}} = et.data

    nt = Nx.backend_transfer(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    assert_raise RuntimeError, ~r"called on deleted or donated buffer", fn ->
      Nx.backend_transfer(et)
    end
  end

  test "Nx.backend_copy/1" do
    t = Nx.tensor([1, 2, 3, 4])

    et = Nx.backend_transfer(t, Beaver.Nx)
    assert %Beaver.Nx{memref: %Exotic.Value.Struct{} = old_buffer} = et.data

    # Copy to the same client/device_id still makes a copy
    et = Nx.backend_copy(t, Beaver.Nx)
    assert %Beaver.Nx{memref: %Exotic.Value.Struct{} = new_buffer} = et.data
    assert old_buffer != new_buffer

    nt = Nx.backend_copy(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>

    nt = Nx.backend_copy(et)
    assert Nx.to_binary(nt) == <<1::64-native, 2::64-native, 3::64-native, 4::64-native>>
  end

  test "Kernel.inspect/2" do
    t = Nx.tensor([1, 2, 3, 4], backend: Beaver.Nx)

    assert inspect(t) ==
             """
             #Nx.Tensor<
               s64[4]
               [1, 2, 3, 4]
             >\
             """
  end
end
