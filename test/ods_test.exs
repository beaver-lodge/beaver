defmodule ODSDumpTest do
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Type
  alias Beaver.MLIR.Dialect.{Func, Arith, MemRef}
  require Func
  use Beaver.Case, async: true

  @moduletag :smoke
  test "lookup" do
    assert {:ok,
            %{
              "attributes" => _,
              "operands" => _,
              "results" => _
            }} = MLIR.ODS.Dump.lookup("affine.for")

    assert {:error, "failed to find ODS dump of \"???\""} = MLIR.ODS.Dump.lookup("???")
  end

  test "enriched ODS facts (traits, regions, builders, class, folder)" do
    assert {:ok, arith_addi} = MLIR.ODS.Dump.lookup("arith.addi")
    assert arith_addi["native_class_name"] == "::mlir::arith::AddIOp"
    assert arith_addi["has_folder"] == true
    assert arith_addi["traits"] |> Enum.any?(&(&1 == "::mlir::OpTrait::IsCommutative"))
    assert arith_addi["traits"] |> Enum.any?(&(&1 == "InferTypeOpInterface"))

    assert {:ok, scf_for} = MLIR.ODS.Dump.lookup("scf.for")
    assert scf_for["regions"] == [%{"name" => "region", "variadic" => false}]
    assert is_list(scf_for["builders"])

    first_builder_parameters =
      scf_for["builders"] |> List.first() |> Map.fetch!("parameters")

    assert Enum.any?(
             first_builder_parameters,
             &(&1["name"] == "lowerBound" and &1["cpp_type"] == "Value")
           )
  end

  test "loads the canonical JSON dump lazily" do
    dump_path = Application.app_dir(:beaver, "priv/generated/ods_dump.json")
    priv_dir = Application.app_dir(:beaver, "priv")

    archived_priv_paths =
      Mix.Project.config()[:make_precompiler_priv_paths]
      |> Enum.flat_map(&Path.wildcard(Path.join(priv_dir, &1)))

    assert MLIR.ODS.Dump.__info__(:attributes)[:external_resource] == [dump_path]
    assert dump_path in archived_priv_paths
    assert :ok = MLIR.ODS.Dump.clear_cache()
    assert {:ok, %{"name" => "affine.for"}} = MLIR.ODS.Dump.lookup("affine.for")
  end

  test "tagged operands", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        Func.func _(
                    function_type: Type.function([Type.i32(), Type.i32()], []),
                    sym_name:
                      Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                  ) do
          region do
            block _(a >>> Type.i32(), b >>> Type.i32()) do
              Arith.addi(lhs: a, rhs: b) >>> Type.i32()
              Func.return() >>> []
            end
          end
        end
      end
      |> MLIR.verify!()
    end
  end

  test "tagged and untagged operands mixed", %{ctx: ctx} do
    mlir ctx: ctx do
      assert_raise ArgumentError, ~r"Cannot mix tagged and untagged operands", fn ->
        mlir ctx: ctx do
          module do
            Func.func _(
                        function_type: Type.function([Type.i32(), Type.i32()], []),
                        sym_name:
                          Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                      ) do
              region do
                block _(a >>> Type.i32(), b >>> Type.i32()) do
                  Arith.addi(a, rhs: b) >>> Type.i32()
                  Func.return() >>> []
                end
              end
            end
          end
        end
      end
    end
  end

  test "tagged operands unconsumed", %{ctx: ctx} do
    import ExUnit.CaptureLog

    mlir ctx: ctx do
      logs =
        capture_log(fn ->
          mlir ctx: ctx do
            module do
              Func.func _(
                          function_type: Type.function([Type.i32(), Type.i32()], []),
                          sym_name:
                            Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                        ) do
                region do
                  block _(_a >>> Type.i32(), b >>> Type.i32()) do
                    Arith.addi(rhs: b) >>> Type.i32()
                    Func.return() >>> []
                  end
                end
              end
            end
          end
        end)

      assert logs =~ ~r"Single operand 'lhs' not set when creating operation arith\.addi"
    end
  end

  test "segment_sizes without ods dump", %{ctx: ctx} do
    mlir ctx: ctx do
      MLIR.Context.allow_unregistered_dialects(ctx)

      m =
        module do
          a = Arith.constant(value: Attribute.integer(Type.i32(), 1)) >>> :infer
          b = Arith.constant(value: Attribute.integer(Type.i32(), 2)) >>> :infer
          UndefinedDialect.foo(lhs: a, rhs: b, operand_segment_sizes: :infer) >>> Type.i32()

          UndefinedDialect.foo(rhs: [a], lhs: [a, b], operand_segment_sizes: :infer) >>>
            Type.i32()
        end

      ops = MLIR.Module.body(m) |> Beaver.Walker.operations()
      assert 1 = ops[2][:operand_segment_sizes][0]
      assert 1 = ops[2][:operand_segment_sizes][1]
      assert 1 = ops[3][:operand_segment_sizes][0]
      assert 2 = ops[3][:operand_segment_sizes][1]
    end
  end

  test "segment_sizes infer in non-generating usage", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        Func.func _(
                    function_type: Type.function([Type.i32(), Type.i32()], []),
                    sym_name:
                      Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                  ) do
          region do
            block _(a >>> Type.i32(), b >>> Type.i32()) do
              Arith.addi(lhs: a, rhs: b) >>> Type.i32()
              Func.return() >>> []
            end
          end
        end
      end
      |> MLIR.verify!()
    end
  end

  test "all-zero segment_sizes", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        Func.func _(
                    function_type: Type.function([], []),
                    sym_name:
                      Beaver.MLIR.Attribute.string("f#{System.unique_integer([:positive])}")
                  ) do
          region do
            block _() do
              MemRef.alloca(operand_segment_sizes: :infer) >>>
                Type.memref!([1], Type.i32(ctx: ctx))

              Func.return() >>> []
            end
          end
        end
      end
      |> MLIR.verify!()
    end
  end
end
