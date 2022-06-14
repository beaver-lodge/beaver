defmodule Exotic.Type.Struct do
  defmacro __using__(fields: fields) do
    quote bind_quoted: [fields: fields] do
      @enforce_keys [:ref]
      defstruct ref: nil, holdings: MapSet.new()

      @type t :: %__MODULE__{
              ref: reference(),
              holdings: MapSet.t(reference())
            }
      Module.register_attribute(__MODULE__, :native_fields, accumulate: false, persist: true)

      @native_fields fields
      defmodule Closures do
        @moduledoc false
        @callback handle_invoke(term(), term(), term()) :: term()
      end

      def native_fields() do
        for {n, ref} <- native_fields_with_names() do
          ref
        end
      end

      def native_type() do
        {:struct, native_fields()}
      end

      def native_fields_with_names() do
        for {n, f} <- @native_fields do
          {n, Exotic.Type.get(f)}
        end
      end
    end
  end
end

defmodule Exotic.Type do
  @moduledoc """
  `Exotic.Type.get/1` will create an internal reprensentation of the type (a rust `Enum`) but not really a libffi type pointer.
  Although this introduces some overhead, it will create new resource before calling C function.
  It makes the call much safer because we don't need to worry about BEAM collecting it.
  In rust code it creates new libffi type from the rust `Enum`. libffi type pointers are not thread safe types in Rust so we can't register them as resource.
  """
  @enforce_keys [:t]
  defstruct [:ref, :t]

  def get(t = :void) do
    %__MODULE__{ref: Exotic.NIF.get_void_type(), t: t}
  end

  def get(t = :ptr) do
    %__MODULE__{ref: Exotic.NIF.get_ptr_type(), t: t}
  end

  def get(t = :size) do
    %__MODULE__{ref: Exotic.NIF.get_size_type(), t: t}
  end

  def get(t = :isize) do
    %__MODULE__{ref: Exotic.NIF.get_size_type(), t: t}
  end

  def get(t = :u8) do
    %__MODULE__{ref: Exotic.NIF.get_u8_type(), t: t}
  end

  def get(t = :u32) do
    %__MODULE__{ref: Exotic.NIF.get_u32_type(), t: t}
  end

  def get(t = :bool) do
    %__MODULE__{ref: Exotic.NIF.get_bool_type(), t: t}
  end

  def get(t = :i8) do
    %__MODULE__{ref: Exotic.NIF.get_i8_type(), t: t}
  end

  def get(t = {:i, 8}) do
    %__MODULE__{ref: Exotic.NIF.get_i8_type(), t: t}
  end

  def get(t = :i32) do
    %__MODULE__{ref: Exotic.NIF.get_i32_type(), t: t}
  end

  def get(t = {:i, 16}) do
    %__MODULE__{ref: Exotic.NIF.get_i16_type(), t: t}
  end

  def get(t = {:i, 32}) do
    %__MODULE__{ref: Exotic.NIF.get_i32_type(), t: t}
  end

  def get(t = :long) do
    %__MODULE__{ref: Exotic.NIF.get_i32_type(), t: t}
  end

  def get(t = {:i, 64}) do
    %__MODULE__{ref: Exotic.NIF.get_i64_type(), t: t}
  end

  def get(t = :i64) do
    %__MODULE__{ref: Exotic.NIF.get_i64_type(), t: t}
  end

  def get(t = :f32) do
    %__MODULE__{ref: Exotic.NIF.get_f32_type(), t: t}
  end

  def get(t = :f64) do
    %__MODULE__{ref: Exotic.NIF.get_f64_type(), t: t}
  end

  def get(t = {:f, 32}) do
    %__MODULE__{ref: Exotic.NIF.get_f32_type(), t: t}
  end

  def get(t = {:f, 64}) do
    %__MODULE__{ref: Exotic.NIF.get_f64_type(), t: t}
  end

  def get(t = {:u, size}) do
    case size do
      8 ->
        %__MODULE__{ref: Exotic.NIF.get_u8_type(), t: t}

      16 ->
        %__MODULE__{ref: Exotic.NIF.get_u16_type(), t: t}

      32 ->
        %__MODULE__{ref: Exotic.NIF.get_u32_type(), t: t}

      64 ->
        %__MODULE__{ref: Exotic.NIF.get_u64_type(), t: t}
    end
  end

  def get({:type_def, module}) do
    case apply(module, :native_type, []) do
      {:struct, native_fields} ->
        native_fields = native_fields |> Enum.map(&Map.get(&1, :ref))

        %__MODULE__{
          ref: Exotic.NIF.get_struct_type(native_fields),
          t: module
        }

      {:ptr, _} ->
        %__MODULE__{
          ref: Exotic.NIF.get_ptr_type(),
          t: module
        }

      enum_type ->
        get(enum_type)
    end
  end

  def get({:ptr, t}) do
    %__MODULE__{ref: Exotic.NIF.get_ptr_type(), t: t}
  end

  def get(module) when is_atom(module) and not is_nil(module) do
    get({:type_def, module})
  end

  def get(%__MODULE__{ref: ref} = this) when not is_nil(ref) do
    this
  end

  # TODO: add docs on struct type
  def get(%__MODULE__{t: module}) when is_atom(module) do
    get(module)
  end

  defmodule Array do
    alias Exotic.{Type, NIF}
    defstruct [:t, :size]

    # use a struct to represent a sized Array
    def get(%Type{ref: ref, t: t}, size) do
      struct_ref = ref |> List.duplicate(size) |> NIF.get_struct_type()
      %Type{ref: struct_ref, t: %__MODULE__{t: t, size: size}}
    end
  end
end
