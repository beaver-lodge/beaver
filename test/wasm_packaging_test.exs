defmodule WasmPackagingTest do
  use Beaver
  use Beaver.Case, async: true

  alias Beaver.MLIR
  alias Beaver.Wasm
  alias MLIR.Dialect.{Func, Arith}
  alias MLIR.Type
  require Func

  defp add_module(ctx) do
    mlir ctx: ctx do
      module do
        Func.func add(
                    function_type: Type.function([Type.i32(), Type.i32()], [Type.i32()]),
                    sym_name: MLIR.Attribute.string("add")
                  ) do
          region do
            block _(a >>> Type.i32(), b >>> Type.i32()) do
              r = Arith.addi(a, b) >>> Type.i32()
              Func.return(r) >>> []
            end
          end
        end
      end
    end
  end

  test "packages a module into a wasm binary with exports manifest", %{ctx: ctx} do
    binary = add_module(ctx) |> Wasm.package_binary!(entry: "add")

    assert binary.bytes |> binary_part(0, 4) == <<0x00, 0x61, 0x73, 0x6D>>
    assert binary.target == "wasm32-wasi"
    assert Enum.any?(binary.exports, &(&1.name == "add" and &1.kind == :func))
    assert binary.imports == []
  end

  test "parses the imports manifest (WASI imports without -nostdlib)", %{ctx: ctx} do
    binary = add_module(ctx) |> Wasm.package_binary!(entry: "add", nostdlib: false)

    assert Enum.any?(
             binary.imports,
             &(&1.module == "wasi_snapshot_preview1" and &1.kind == :func)
           )
  end

  test "runs the packaged binary in node", %{ctx: ctx} do
    binary = add_module(ctx) |> Wasm.package_binary!(entry: "add")

    wasm_path =
      Path.join(System.tmp_dir!(), "beaver_wasm_#{System.unique_integer([:positive])}.wasm")

    File.write!(wasm_path, binary.bytes)

    on_exit(fn -> File.rm(wasm_path) end)

    {output, 0} =
      System.cmd(
        "node",
        [
          "-e",
          """
          const fs = require('fs');
          const buf = fs.readFileSync(process.argv[1]);
          WebAssembly.instantiate(buf, {}).then(r => console.log(r.instance.exports.add(2, 3)));
          """,
          wasm_path
        ], stderr_to_stdout: true)

    assert String.trim(output) == "5"
  end
end
