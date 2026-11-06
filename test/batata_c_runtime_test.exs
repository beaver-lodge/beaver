defmodule BatataCRuntimeTest do
  use Beaver.Case, async: true
  alias Beaver.{MLIR, Native}
  alias Beaver.MLIR.Conversion.{Ex, Plan}
  alias Beaver.MLIR.Conversion.Ex.CRuntime
  @fixture Path.expand("fixtures/batata_c", __DIR__)
  @namespace "abi_fixture"

  setup %{ctx: ctx} do
    Beaver.Slang.load(ctx, Beaver.MLIR.Dialect.Ex)
    :ok
  end

  @tag :batata_runtime
  test "admitted Batata values, chunked imports and cross-process copies execute through MLIR", %{
    ctx: ctx
  } do
    for mode <- [:raw, :values],
        do:
          check_bridge!(
            ctx,
            System.fetch_env!("BATATA_TEST_LIBRARY"),
            System.fetch_env!("BATATA_TEST_NAMESPACE"),
            mode,
            "beaver_batata_transactions"
          )
  end

  @tag :tmp_dir
  test "MLIR calls the C ABI with owned handles, bounded stepping and complete cleanup", %{
    ctx: ctx,
    tmp_dir: dir
  } do
    libraries =
      for namespace <- [@namespace, "another_fixture"] do
        adapter = Path.join(dir, namespace <> ".c")
        File.write!(adapter, CRuntime.adapter_source(namespace))
        library = Path.join(dir, namespace <> extension())

        args = [
          "cc",
          "-shared",
          "-fPIC",
          "-fvisibility=hidden",
          ~s(-DBTT_TEST_FIXTURE_ID="#{namespace}"),
          "-I",
          @fixture,
          adapter,
          Path.join(@fixture, "provider.c"),
          Path.join(@fixture, "aggregates.c"),
          "-o",
          library
        ]

        assert {_, 0} = System.cmd("zig", args, stderr_to_stdout: true)
        {library, namespace}
      end

    for {library, namespace} <- libraries ++ [hd(libraries)], mode <- [:raw, :values] do
      check_bridge!(
        ctx,
        library,
        namespace,
        mode,
        "beaver_batata_aggregates"
      )
    end
  end

  @tag :tmp_dir
  test "adapter compilation rejects legacy magic and mismatched ABI versions", %{tmp_dir: dir} do
    adapter = Path.join(dir, "adapter.c")
    File.cp!(Path.join(@fixture, "api.def"), Path.join(dir, "api.def"))
    File.write!(adapter, CRuntime.adapter_source(@namespace))

    for abi <- ["0x44414f540000000b", "0x44414f540000000c", "0x425454000000000b"] do
      File.write!(
        Path.join(dir, "aot.h"),
        File.read!(Path.join(@fixture, "aot.h"))
        |> String.replace("0x425454000000000c", abi)
      )

      {output, status} =
        System.cmd("zig", ["cc", "-c", "-I", dir, adapter, "-o", Path.join(dir, "adapter.o")],
          stderr_to_stdout: true
        )

      assert status != 0
      assert output =~ "unsupported Batata C ABI"
    end
  end

  test "the public header and Ex schema expose the BTT namespace without legacy aliases" do
    header = File.read!(Path.join(@fixture, "aot.h"))

    functions =
      Regex.scan(~r/^BTT_TEST_CALL\((\w+),/m, File.read!(Path.join(@fixture, "api.def")),
        capture: :all_but_first
      )
      |> List.flatten()
      |> Enum.sort()

    assert functions == CRuntime.functions()
    assert length(functions) == 49
    assert CRuntime.abi_id() == 0x425454000000000C
    refute header =~ ~r/\b(?:DA_|Da[A-Z]|da_)/
    refute CRuntime.adapter_source(@namespace) =~ ~r/\b(?:DA_|Da[A-Z]|da_)/
    refute Enum.any?(Beaver.MLIR.Dialect.Ex.operations(), &String.starts_with?(&1, "ex.da_"))
  end

  test "lowering reuses matching declarations and rejects conflicting symbols", %{ctx: ctx} do
    for {type, succeeds} <- [{"i64", true}, {"i32", false}] do
      module =
        parse!(ctx, """
        module {
          func.func private @beaver_batata_abi_fixture_query(#{type}, !llvm.ptr) -> i32
          func.func @query(%abi: i64, %out: !llvm.ptr) -> i32 {
            %a = "ex.btt_query"(%abi, %out) : (i64, !llvm.ptr) -> i32
            %b = "ex.btt_query"(%abi, %out) : (i64, !llvm.ptr) -> i32
            return %b : i32
          }
        }
        """)

      if succeeds do
        assert {:ok, _, _} = Plan.run(Ex.plan(runtime_namespace: @namespace), module)

        assert length(
                 Regex.scan(
                   ~r/func.func private @beaver_batata_abi_fixture_query/,
                   MLIR.to_string(module)
                 )
               ) == 1
      else
        assert {:error, %MLIR.Conversion.Error{}} =
                 Plan.run(Ex.plan(runtime_namespace: @namespace), module)
      end
    end
  end

  test "C ABI operation signatures and output storage round-trip", %{ctx: ctx} do
    for name <- CRuntime.functions(), mode <- [:raw, :values] do
      module = parse!(ctx, function(name, mode))

      roundtrip =
        module |> MLIR.Bytecode.write!() |> MLIR.Module.create!(ctx: ctx) |> MLIR.verify!()

      assert MLIR.to_string(roundtrip) == MLIR.to_string(module)
      lowered = Plan.run!(Ex.plan(runtime_namespace: @namespace), roundtrip) |> MLIR.verify!()
      refute MLIR.to_string(lowered) =~ "!ex."
    end

    for storage <- ~w(runtime entry process value query step view event import copy callable)a do
      type = CRuntime.storage_type(storage, ctx: ctx)
      refute MLIR.null?(type)
    end

    assert_raise ArgumentError, fn ->
      parse!(
        ctx,
        ~s[func.func @bad(%r: !llvm.ptr, %n: i64, %out: !llvm.ptr) -> i32 { %s = "ex.btt_step"(%r, %n, %out) : (!llvm.ptr, i64, !llvm.ptr) -> i32 return %s : i32 }]
      )
    end
  end

  test "runtime namespaces are required, validated and included in plan receipts", %{ctx: ctx} do
    for namespace <- [nil, "", "bad-name", String.duplicate("a", 129)] do
      assert_raise ArgumentError, ~r/runtime_namespace/, fn ->
        CRuntime.adapter_source(namespace)
      end
    end

    refute Plan.declaration(Ex.plan(runtime_namespace: "first")) ==
             Plan.declaration(Ex.plan(runtime_namespace: "second"))

    module = parse!(ctx, function("query"))
    assert {:error, %MLIR.Conversion.Error{}} = Plan.run(Ex.plan(), module)
  end

  defp check_bridge!(ctx, library, namespace, mode, driver) do
    calls = CRuntime.functions()

    module =
      parse!(
        ctx,
        "module {\n" <>
          Enum.map_join(calls, "\n", &function(&1, mode)) <>
          "\n" <> probe(namespace, calls, driver) <> "\n}"
      )

    module = Plan.run!(Ex.plan(runtime_namespace: namespace), module) |> MLIR.verify!()
    module = module |> Beaver.Composer.append("convert-func-to-llvm") |> Beaver.Composer.run!()
    jit = MLIR.ExecutionEngine.create!(module, shared_lib_paths: [library])

    try do
      result = Native.I32.make(0)
      MLIR.ExecutionEngine.invoke!(jit, "probe", [], result)

      assert Native.to_term(result) == 0,
             "#{mode} #{driver} failed at C source line #{Native.to_term(result)}"
    after
      MLIR.ExecutionEngine.destroy(jit)
    end
  end

  defp function(name, mode \\ :raw) do
    types = CRuntime.argument_types(name)
    args = types |> Enum.with_index() |> Enum.map_join(", ", fn {t, i} -> "%a#{i}: #{t}" end)
    operands = types |> Enum.with_index() |> Enum.map_join(", ", fn {_, i} -> "%a#{i}" end)

    body =
      if mode == :raw do
        ~s[%status = "ex.btt_#{name}"(#{operands}) : (#{Enum.join(types, ", ")}) -> i32]
      else
        value_function_body(name)
      end

    "llvm.func @jit_#{name}(#{args}) -> i32 { #{body}\nllvm.return %status : i32 }"
  end

  defp value_function_body(name) do
    params = Enum.with_index(CRuntime.value_parameters(name))

    loads =
      for {{:aggregate, type}, i} <- params,
          do: ~s[%v#{i} = "ex.value_load"(%a#{i}) : (!llvm.ptr) -> #{type}]

    operands =
      for {{direction, _}, i} <- params,
          direction != :out,
          do: if(direction == :aggregate, do: "%v#{i}", else: "%a#{i}")

    outputs = for {{:out, type}, i} <- params, do: {"%v#{i}", type, i}
    bindings = ["%status" | Enum.map(outputs, &elem(&1, 0))]
    {inputs, results} = CRuntime.value_signature(name)

    call =
      ~s[#{Enum.join(bindings, ", ")} = "ex.value_#{name}"(#{Enum.join(operands, ", ")}) : (#{Enum.join(inputs, ", ")}) -> (#{Enum.join(results, ", ")})]

    stores =
      for {value, type, i} <- outputs do
        if String.starts_with?(type, "!ex."),
          do: ~s["ex.value_store"(#{value}, %a#{i}) : (#{type}, !llvm.ptr) -> ()],
          else: ~s[llvm.store #{value}, %a#{i} : #{type}, !llvm.ptr]
      end

    guards =
      params
      |> Enum.filter(fn {{direction, _}, _} -> direction == :aggregate end)
      |> Enum.map(fn {_, i} -> ~s[%null#{i} = llvm.mlir.zero : !llvm.ptr
      %bad#{i} = llvm.icmp "eq" %a#{i}, %null#{i} : !llvm.ptr
      llvm.cond_br %bad#{i}, ^invalid, ^valid#{i}
      ^valid#{i}:] end)

    body = Enum.join(guards ++ loads ++ [call] ++ stores, "\n")

    if guards == [],
      do: body,
      else:
        String.replace(body, "%status", "%call_status") <>
          "\nllvm.br ^exit(%call_status : i32)\n^invalid:\n%invalid_status = llvm.mlir.constant(1 : i32) : i32\nllvm.br ^exit(%invalid_status : i32)\n^exit(%status: i32):"
  end

  defp probe(namespace, calls, driver) do
    slots =
      calls
      |> Enum.with_index()
      |> Enum.map_join("\n", fn {name, i} ->
        "%f#{i} = llvm.mlir.addressof @jit_#{name} : !llvm.ptr\n%p#{i} = llvm.getelementptr %table[#{i}] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr\nllvm.store %f#{i}, %p#{i} : !llvm.ptr, !llvm.ptr"
      end)

    """
    llvm.func @#{driver}(!llvm.ptr, !llvm.ptr) -> i32
    llvm.mlir.global private constant @expected_identity("#{namespace}\\00")
    func.func @probe() -> i32 attributes {llvm.emit_c_interface} {
      %count = llvm.mlir.constant(#{length(calls)} : i64) : i64
      %table = llvm.alloca %count x !llvm.ptr : (i64) -> !llvm.ptr
      #{slots}
      %identity = llvm.mlir.addressof @expected_identity : !llvm.ptr
      %result = llvm.call @#{driver}(%table, %identity) : (!llvm.ptr, !llvm.ptr) -> i32
      return %result : i32
    }
    """
  end

  defp extension do
    case :os.type() do
      {:win32, _} -> ".dll"
      {:unix, :darwin} -> ".dylib"
      _ -> ".so"
    end
  end

  defp parse!(ctx, source), do: MLIR.Module.create!(source, ctx: ctx) |> MLIR.verify!()
end
