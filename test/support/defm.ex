defmodule TranslateMLIR do
  defmacro __using__(_) do
    quote do
      import TranslateMLIR
    end
  end

  defmacro defm(call, expr \\ nil) do
  end
end
