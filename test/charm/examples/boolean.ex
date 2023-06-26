defmodule MyBoolean do
  alias Charm.Boolean

  def main() do
    t = %Boolean{}

    if t do
      IO.puts("it is true")
    end

    f = %Boolean{value: false}

    if f do
      IO.puts("it is false")
    end

    t = %Boolean{value: true}

    if t do
      IO.puts("it is false")
    end
  end
end
