defmodule ExoticCodeGenTest do
  use ExUnit.Case

  @tag timeout: :infinity, slow: true
  test "code gen" do
    defmodule TestHeaderModule do
    end

    i = %Exotic.Header{
      file: "include/wrapper/llvm/14.h",
      search_paths: [
        Beaver.LLVM.Config.include_dir(),
        Path.join(File.cwd!(), "native/mlir_nif/met/include")
      ],
      module_name: TestHeaderModule
    }

    Exotic.Header.parse(i)
  end

  test "test mlir nif" do
    Beaver.MLIR.NIF.add(1, 2)
    assert Beaver.MLIR.NIF.module_info(:attributes) |> Keyword.get(:load_from)
  end
end
