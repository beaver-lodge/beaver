defmodule Fizz.ResourceKind do
  defmacro __using__(opts) do
    root_module = Keyword.fetch!(opts, :root_module)
    forward_module = Keyword.get(opts, :forward_module, root_module)
    fields = Keyword.get(opts, :fields) || []

    quote bind_quoted: [
            root_module: root_module,
            forward_module: forward_module,
            fields: fields
          ] do
      defstruct [ref: nil, bag: MapSet.new()] ++ fields

      @type t :: %__MODULE__{
              ref: reference(),
              bag: MapSet.t()
            }

      def array(list, opts \\ []) when is_list(list) do
        unquote(forward_module).array(list, __MODULE__, opts)
      end

      def make(value) do
        %__MODULE__{
          ref:
            apply(
              unquote(root_module),
              Module.concat([__MODULE__, "make"]) |> unquote(forward_module).check!(),
              [value]
            )
        }
      end
    end
  end
end
