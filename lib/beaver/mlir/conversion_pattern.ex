defmodule Beaver.MLIR.ConversionPattern do
  @moduledoc """
  Callback-backed dialect conversion patterns.

  `add/5` creates a pattern and immediately transfers its ownership to the
  rewrite pattern set, so a standalone native pattern cannot be leaked.
  """

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  alias Beaver.MLIR

  @type callback() ::
          (MLIR.Operation.t(),
           [MLIR.Value.t()]
           | [[MLIR.Value.t()]],
           MLIR.ConversionPatternRewriter.t() ->
             :ok | :no_match | {:error, term()})

  @spec add(
          MLIR.RewritePatternSet.t(),
          String.Chars.t(),
          MLIR.TypeConverter.t(),
          callback(),
          keyword()
        ) :: MLIR.RewritePatternSet.t()
  def add(
        %MLIR.RewritePatternSet{ref: set_ref} = set,
        root_name,
        %MLIR.TypeConverter{registration: registration},
        callback,
        opts \\ []
      )
      when is_function(callback, 3) do
    case Keyword.keys(opts) -- [:ctx, :benefit, :one_to_n, :timeout] do
      [] ->
        :ok

      unsupported ->
        raise ArgumentError, "unsupported ConversionPattern options: #{inspect(unsupported)}"
    end

    %MLIR.Context{ref: context_ref} =
      Keyword.get(opts, :ctx) || raise ArgumentError, "option :ctx is required"

    benefit = Keyword.get(opts, :benefit, 1)
    one_to_n = Keyword.get(opts, :one_to_n, false)
    timeout_ms = Keyword.get(opts, :timeout, 30_000)

    unless is_integer(benefit) and benefit >= 0 do
      raise ArgumentError, ":benefit must be a non-negative integer"
    end

    unless is_boolean(one_to_n) do
      raise ArgumentError, ":one_to_n must be boolean"
    end

    unless is_integer(timeout_ms) and timeout_ms >= 0 do
      raise ArgumentError, ":timeout must be a non-negative integer"
    end

    :ok =
      MLIR.CAPI.beaver_raw_conversion_pattern_add(
        set_ref,
        to_string(root_name),
        benefit,
        context_ref,
        registration,
        callback,
        one_to_n,
        timeout_ms
      )

    set
  end
end
