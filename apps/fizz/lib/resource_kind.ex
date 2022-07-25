defmodule Fizz.ResourceKind do
  defmacro __using__(opts) do
    zig_t = Keyword.fetch!(opts, :zig_t)
    root_module = Keyword.fetch!(opts, :root_module)
    fields = Keyword.get(opts, :fields) || []

    quote bind_quoted: [root_module: root_module, zig_t: zig_t, fields: fields] do
      defstruct [ref: nil, zig_t: zig_t, bag: MapSet.new()] ++ fields

      @type t :: %__MODULE__{
              ref: reference(),
              zig_t: binary(),
              bag: MapSet.t()
            }
      def zig_t(), do: unquote(zig_t)

      def array(list, opts \\ []) when is_list(list) do
        unquote(root_module).array(list, __MODULE__, opts)
      end

      # TODO: move this once modules like Beaver.Data.F32 are ready
      def memref(
            allocated,
            aligned,
            offset,
            sizes,
            strides
          ) do
        apply(
          unquote(root_module),
          Module.concat([__MODULE__, "create_memref"]) |> unquote(root_module).check!(),
          [allocated, aligned, offset, sizes, strides]
        )
      end

      def make(value) do
        %__MODULE__{
          ref:
            apply(
              unquote(root_module),
              Module.concat([__MODULE__, "make"]) |> unquote(root_module).check!(),
              [value]
            )
        }
      end
    end
  end
end
