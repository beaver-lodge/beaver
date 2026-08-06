defmodule Beaver.MLIR.ConversionConfig do
  @moduledoc "An owning high-level wrapper for MLIR dialect conversion configuration."

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  alias Beaver.MLIR

  @folding_modes %{never: 0, before_patterns: 1, after_patterns: 2}

  @spec create(keyword()) :: t()
  def create(opts \\ []) do
    config = MLIR.CAPI.mlirConversionConfigCreate()

    try do
      Enum.each(opts, fn
        {:folding_mode, mode} ->
          set_folding_mode(config, mode)

        {:build_materializations, enabled} when is_boolean(enabled) ->
          MLIR.CAPI.mlirConversionConfigEnableBuildMaterializations(config, enabled)

        {key, _value} ->
          raise ArgumentError, "unsupported conversion config option: #{inspect(key)}"
      end)

      config
    rescue
      exception ->
        MLIR.CAPI.mlirConversionConfigDestroy(config)
        reraise exception, __STACKTRACE__
    end
  end

  defp set_folding_mode(config, mode) do
    case @folding_modes do
      %{^mode => value} -> MLIR.CAPI.mlirConversionConfigSetFoldingMode(config, value)
      _ -> raise ArgumentError, "unsupported conversion folding mode: #{inspect(mode)}"
    end
  end

  @spec destroy(t()) :: :ok
  defdelegate destroy(config), to: MLIR.CAPI, as: :mlirConversionConfigDestroy

  @spec with(keyword(), (t() -> result)) :: result when result: var
  def with(opts \\ [], fun) when is_function(fun, 1) do
    config = create(opts)

    try do
      fun.(config)
    after
      destroy(config)
    end
  end
end
