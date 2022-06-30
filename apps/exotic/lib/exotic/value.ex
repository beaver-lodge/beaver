# TODO: define Value as Protocol (Is it necessary?)
# Pro: allow user to implement its own struct plug into Exotic
# Con: could be very complicated, struct and array could become exceptions
# TODO: Value.Struct should support Access behavior (Is it necessary?)

# TODO: do we need a struct specific for field?
# fetch/2 in Access behavior returns a Ferrent.Field and use other functions make it a value/ptr
# struct_value[:field_name] |> Exotic.Struct.Field.as_ptr()
# struct_value[:field_name] |> Exotic.Struct.Field.as_value()
# a Array.Element for Array as well
# some_array[1] |> Exotic.Array.Element.as_value()

defmodule Exotic.Element do
  @moduledoc """
  Exotic.Element is a special struct to reperesent a field of a Exotic.Struct or an element of an Exotic.Array.
  It doesn't have real memory to back it when it is created. To get a copy(Value) or a pointer value of the real memory,
  explicitly calling the `as_ptr` method is required.
  """
end

defmodule Exotic.Pointer do
  @moduledoc """
  Exotic.Pointer could be pointer to a Elixir owned Exotic.Value or a pointer to a C owned memory.
  You can only call `as_value` on pointer to a Elixir owned Exotic.Value.
  """

  @doc """
  If this is a pointer to Elixir's owned memory
  """
  def is_owned?() do
  end
end

defprotocol Exotic.Valuable do
  @fallback_to_any true
  @doc """
  return a NIF ValueWrapper resource defined in `native/exotic_nif/src/lib.rs`.
  By implementing this function you can make Exotic works with your own Rust types.
  """
  def resource(data)

  @doc """
  what resources should be transmitted, usually it is the combination of resource itself and other holdings, should return a `MapSet.t()`
  """
  def holdings(data)
  def transmit(data, ref)
  def type(v)
end

defimpl Exotic.Valuable, for: Any do
  def resource(%{ref: ref}), do: ref
  def holdings(%{holdings: holdings, ref: ref}), do: MapSet.put(holdings, ref)

  def transmit(v = %{holdings: holdings}, ref),
    do: struct!(v, %{holdings: MapSet.put(holdings, ref)})

  def type(%{type: type}) do
    type
  end

  def type(%module{}) do
    module
  end
end

defmodule Exotic.Value do
  defmodule Ptr do
    @enforce_keys [:ref, :holdings]
    defstruct ref: nil, holdings: MapSet.new(), type: :ptr

    def null() do
      %Exotic.Value.Ptr{ref: Exotic.NIF.get_null_ptr_value(), holdings: MapSet.new()}
    end

    def read_as_binary(%{ref: ptr_ref}, length) do
      %{ref: length_ref} = Exotic.Value.get(length)
      Exotic.NIF.read_ptr_content_as_binary(ptr_ref, length_ref)
    end
  end

  defmodule I32 do
    @enforce_keys [:ref]
    defstruct ref: nil, holdings: MapSet.new()
  end

  defmodule I64 do
    @enforce_keys [:ref]
    defstruct ref: nil, holdings: MapSet.new()
  end

  alias Exotic.NIF
  @enforce_keys [:holdings, :type]
  defstruct [:ref, :type, holdings: MapSet.new()]

  # To support Access behavior, each struct should have its own elixir struct

  def fetch(%{ref: v_ref, holdings: holdings, fields: fields}, key) do
    fetch(%{ref: v_ref, holdings: holdings}, fields, key)
  end

  def fetch(%{ref: v_ref, holdings: holdings}, module, key) when is_atom(module) do
    fields = apply(module, :native_fields_with_names, [])
    fetch(%{ref: v_ref, holdings: holdings}, fields, key)
  end

  def fetch(%{ref: v_ref, holdings: holdings}, fields, key) when is_list(fields) do
    %Exotic.Type{ref: t_ref} = Exotic.Type.get({:struct, fields})

    fields =
      for {n, f} <- fields do
        {n, Exotic.Type.get(f)}
      end

    i = Enum.find_index(fields, fn {n, _f} -> n == key end)
    {_n, %Exotic.Type{ref: _, t: field_t}} = Enum.find(fields, fn {n, _f} -> n == key end)
    res_ref = Exotic.NIF.access_struct_field_as_value(t_ref, v_ref, i)
    # although it is value semantic, still inherent transmits, to make it safer
    %__MODULE__{
      ref: res_ref,
      type: %Exotic.Type{ref: nil, t: field_t},
      holdings: MapSet.put(holdings, v_ref)
    }
  end

  @doc """
  If this is a pointer to Elixir owned memory, the Exotic.Value this function returns should have the same lifetime as the argument Exotic.Value.
  """
  def get_ptr(data) do
    ptr = Exotic.Valuable.resource(data) |> Exotic.NIF.get_ptr()
    holdings = Exotic.Valuable.holdings(data)
    %Exotic.Value.Ptr{ref: ptr, holdings: holdings}
  end

  def as_ptr(data) do
    ptr = Exotic.Valuable.resource(data) |> Exotic.NIF.as_ptr()
    holdings = Exotic.Valuable.holdings(data)
    %Exotic.Value.Ptr{ref: ptr, holdings: holdings}
  end

  def as_binary(data) do
    Exotic.Valuable.resource(data) |> Exotic.NIF.as_binary()
  end

  def transmit(v = %{ref: ref, holdings: holdings}) do
    struct!(v, %{holdings: MapSet.put(holdings, ref)})
  end

  @doc """
  precendence:
  - explicit type in value struct
  - explicit type in function definition
  - default types for erlang term
  """
  def get(v) when is_binary(v) do
    %__MODULE__{
      ref: NIF.get_c_string_value(v),
      holdings: MapSet.new(),
      type: :ptr
    }
    |> Exotic.Value.get_ptr()
  end

  def get(v) when is_integer(v) do
    %__MODULE__{
      ref: NIF.get_i32_value(v),
      holdings: MapSet.new(),
      type: :i32
    }
  end

  def get(v) when is_float(v) do
    %__MODULE__{
      ref: NIF.get_f64_value(v),
      holdings: MapSet.new(),
      type: :f64
    }
  end

  def get(true) do
    %__MODULE__{
      ref: NIF.get_i8_value(1),
      holdings: MapSet.new(),
      type: :f64
    }
  end

  def get(false) do
    %__MODULE__{
      ref: NIF.get_i8_value(0),
      holdings: MapSet.new(),
      type: :f64
    }
  end

  def get(v) do
    v
  end

  # TODO: add isize NIF
  def get(t = :isize, v) when is_integer(v) do
    %__MODULE__{
      ref: NIF.get_i64_value(v),
      holdings: MapSet.new(),
      type: t
    }
  end

  @doc """
  create a closure ptr with a function type.
  If you want to use one process to handle multiple kinds of callbacks, you should use get/3 to provide a callback_id.
  """
  def get(
        t = %Exotic.Type{
          ref: _,
          t: [{:function, [_ret, _args_kv]}]
        },
        value
      )
      when is_atom(value) do
    get_closure(t, value, :invoke_callback)
  end

  def get({:type_def, _module}, value) do
    value
  end

  def get({:ptr, [:void]}, value) do
    get(value)
  end

  def get(:bool, value) when is_boolean(value), do: get(value)
  def get({:i, 32}, value) when is_integer(value), do: get(value)

  def get(:f32, value) when is_float(value) do
    %__MODULE__{
      ref: NIF.get_f32_value(value),
      holdings: MapSet.new(),
      type: :f32
    }
  end

  def get(:i64, value) when is_integer(value) do
    %__MODULE__{
      ref: NIF.get_i64_value(value),
      holdings: MapSet.new(),
      type: :i64
    }
  end

  def get({:u, 32}, value) when is_integer(value) do
    %__MODULE__{
      ref: NIF.get_u32_value(value),
      holdings: MapSet.new(),
      type: :u32
    }
  end

  def get(_t, %Exotic.Value{} = v) do
    v
  end

  def get_closure(
        %Exotic.Type{
          ref: _,
          t: [{:function, [ret | args_kv]}]
        },
        value,
        callback_id \\ :invoke_callback
      )
      when is_atom(value) do
    args_types = args_kv |> Enum.map(fn type -> Exotic.Type.get(type) end)
    ret_type = Exotic.Type.get(ret)

    Exotic.Closure.create(
      %Exotic.Closure.Definition{arg_types: args_types, return_type: ret_type},
      value,
      callback_id
    )
    |> Exotic.Value.as_ptr()
  end

  def get_values_by_types(values, types) do
    Enum.zip(types, values)
    |> Enum.map(fn {t, v} -> __MODULE__.get(t, v) end)
  end

  def extract(%__MODULE__{ref: nil}) do
    raise "Cannot extract value because it has a nif ref"
  end

  def extract(%{ref: ref}) do
    ref |> Exotic.NIF.extract()
  end

  defmodule String do
    alias Exotic.Value

    def get(v) when is_binary(v) do
      %Value{ref: NIF.get_c_string_value(v), type: :ptr, holdings: MapSet.new()}
    end

    def extract(value, len) do
      data_ref = value |> Exotic.Valuable.resource()
      length_ref = len |> Exotic.Valuable.resource()
      Exotic.NIF.extract_c_string_as_binary_string(data_ref, length_ref)
    end
  end

  defmodule Char do
    # this is for erlang char list
    def get(v) when is_integer(v) do
    end
  end

  defmodule Struct do
    @moduledoc """
    Create and extract C structs.
    """
    alias Exotic.{Value, Type}
    defstruct ref: nil, holdings: MapSet.new(), type: :struct, fields: :undefined

    def extract(module, %Exotic.Value{ref: ref}) when is_atom(module) do
      struct_t =
        apply(module, :native_fields, [])
        |> Enum.map(&Exotic.Type.get/1)
        |> Enum.map(&Map.fetch!(&1, :ref))
        |> Exotic.NIF.get_struct_type()

      Exotic.NIF.extract_struct(struct_t, ref)
    end

    def extract(struct_t, %Exotic.Value{ref: ref}) when is_reference(struct_t) do
      Exotic.NIF.extract_struct(struct_t, ref)
    end

    defp get_values_by_types_and_names(types, values) do
      for {{name, type}, value} <- Enum.zip(types, values) do
        case {type, value} do
          {%Exotic.Type{
             ref: _,
             t: [{:function, [_ret | _args_kv]}]
           }, _} ->
            Exotic.Value.get_closure(type, value, name)

          {%Exotic.Type{}, v = %{ref: _ref}} ->
            v

          {%Exotic.Type{ref: _ref, t: :i64}, i} when is_integer(i) ->
            ref = Exotic.NIF.get_i64_value(i)
            %Exotic.Value.I64{ref: ref, holdings: MapSet.new()}

          _ ->
            __MODULE__.get(type, value)
        end
      end
    end

    @doc """
    If it is a function pointer, you should pass callback handler module or function as the correspondent value.
    When passing pid, the callback_id will be the struct field name.
    # TODO: support passing function
    """
    def get(module, values) when is_atom(module) do
      types = apply(module, :native_fields_with_names, [])
      get(types, values)
    end

    def get([], _), do: raise("Cannot create struct with no fields")
    def get(_, []), do: raise("Cannot create struct with no values")

    def get(types, values) when is_list(types) do
      types =
        for {n, f} <- types do
          {n, Exotic.Type.get(f)}
        end

      type_refs = types |> Enum.map(fn {_, t} -> t end) |> Enum.map(&Map.fetch!(&1, :ref))

      values = get_values_by_types_and_names(types, values)
      value_refs = values |> Enum.map(&Map.fetch!(&1, :ref))

      ref = NIF.get_struct_value(type_refs, value_refs)

      holdings =
        values
        |> Enum.map(&Map.get(&1, :holdings))
        |> Enum.reduce(MapSet.new(), &MapSet.union(&2, &1))

      struct!(__MODULE__, %{
        ref: ref,
        holdings: holdings,
        fields: types
      })
    end

    def get(binary) when is_binary(binary),
      do:
        struct!(__MODULE__, %{
          ref: Exotic.NIF.get_struct_value_from_binary(binary),
          holdings: MapSet.new()
        })
  end

  defmodule Array do
    alias Exotic.{Value, Type}

    # allocate a struct and copy elements to it
    # this array must outlive the pointer created from it
    def get(values) when length(values) > 0 do
      values = values |> Enum.map(&Value.get/1)

      types = values |> Enum.map(&Exotic.Valuable.type/1) |> Enum.map(&Type.get/1)

      type = List.first(types)
      size = length(types)
      type_refs = types |> Enum.map(&Map.fetch!(&1, :ref))
      value_refs = values |> Enum.map(&Map.fetch!(&1, :ref))

      holdings =
        values
        |> Enum.map(&Map.get(&1, :holdings))
        |> Enum.reduce(MapSet.new(), &MapSet.union(&2, &1))

      # TODO: check values has same types
      ref = NIF.get_struct_value(type_refs, value_refs)

      %Value{
        ref: ref,
        type: Type.Array.get(type, size),
        holdings: holdings
      }
    end

    def get([]) do
      0 |> Exotic.Value.get() |> Exotic.Value.as_ptr()
    end

    # TODO: if length(values) == 0, requires a type
  end
end
