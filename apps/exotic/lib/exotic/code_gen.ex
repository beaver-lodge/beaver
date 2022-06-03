# TODO: use clang's `-ast-dump=json` and run clang in another process to make it safer
defmodule Exotic.CodeGen do
  use Rustler, otp_app: :exotic, crate: "exotic_codegen"

  def parse_header(_header), do: :erlang.nif_error(:nif_not_loaded)

  defmodule Function do
    defstruct name: nil, args: [], ret: nil
  end

  defmodule Struct do
    defstruct name: nil, fields: []
  end

  defmodule TypeDef do
    defstruct name: nil, ty: nil
  end
end
