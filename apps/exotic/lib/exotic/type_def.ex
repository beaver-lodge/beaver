defmodule Exotic.TypeDef do
  defmacro __using__(as: as) do
    # TODO: support custom rules for disable docs for modules
    quote do
      Module.register_attribute(__MODULE__, :alias, accumulate: false, persist: true)
      @alias unquote(Macro.escape(as))
      # TODO: native_type should be a callback in Behavior TypeProvider or TypeDef
      def native_type() do
        @alias
      end
    end
  end
end
