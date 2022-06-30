# Exotic 👽

Elixir binding to libffi built on top of Rust binding to libffi, with the help of Rustler. It has these major features:

- Generate Elixir functions from C header
- Loading C library and call C functions by symbol
- Mechanism for creating and managing data of C types (primitive, struct, array, etc.)

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `exotic` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:exotic, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/exotic>.

## Getting started

### `Valuable` protocol and `TypeDef` behavior

- `Valuable` protocol defines how a Elixir struct should interact with C data.
- `Exotic.Type.Definition` behavior has callbacks describing what a C type is like.

Exotic ships with `Valuable`s implementation of basic C types' like int64, float64, char, etc. Usually you need to implement `Valuable` for your own C struct types or you can add `@include` attributes to have Exotic generate it if you have the header.

It is designed to decouple the interpretation and behavior so that it is possible to combine and customize them wherever appropriate.

<!-- TODO -->

## Should I use Rustler or Exotic?

Exotic is actually built on top of Rustler, so you are using Rustler when you are using Exotic.
In NIF implemented with Rustler you can use or create Exotic types and values.
Consider using Exotic in these cases:

- There is no `-sys` crate in Rust for the C library you are trying to use.
- You only want to write pure Elixir and for whatever reason you don't want to use Rust.
- You need to create, read C struct or array dynamically, instead of creating a lot of NIFs for it.
- The C library's API surface is so large that there is too much work to cover them all.

Note: you can also use a `-sys` crate as a packaging mechanism only but use Exotic to call it without building your own NIFs.

<!-- TODO: should we name value ARC? -->

## How it works under the hood

Exotic works like a mini compiler embedded in Elixir so it has its own concepts of types and values.

- You can create a `Exotic.Value` by calling a native function via `Exotic.call`,
  or create it by helper Functions like `Exotic.Value.String.get("foo")`.
- You can create a `Exotic.Type` by calling a helper function like `Exotic.Type.get(:i32)`.
- When you call a function from Elixir, `Exotic.Type`s will be used to generated libffi's type representation and
  memory `Exotic.Value`s hold will be passed as pointer to libffi function call.

### Convention

APIs in this library should follow these convention:

- If it is a `create` function, user should manage the resource and call `destroy` to free it. Usually there will be a process pool keeping track of this kind of resources.
- If it is a `get` function, user doesn't need to manage the resource and can leave it out there to be collected by BEAM GC.
- It is advised to follow these conventions in higher level libraries.

## Working with/~~against~~ Erlang's GC

If a Erlang process uses little memory, the garbage collector can be very aggressive and responsive.
To address this we need some special measures to make sure a erlang NIF resource will not be collected before a C function really start loading/modifying the data. To learn more about Erlang GC, please read [this](https://github.com/erlang/otp/blob/master/erts/emulator/internal_doc/GarbageCollection.md)

### Value mechanism

`Exotic.Value` is a Elixir Struct holds NIF resource reference to the memory allocated when calling the C library.
If we always transparently translate values to Erlang/Elixir, the memory will be deallocated before we want to use it in later C function calls
because Erlang's GC is very responsive.
In other words, `Exotic.Value` lets Elixir own the data.

Also `Exotic.Value` also make it possible to introduce mutability in Elixir.
You can create a `Exotic.Value` and pass it as a pointer to C function:

```elixir
v = 2022.3 |> Exotic.Value.get()
v |> Exotic.get_ptr() |> some_c_func()
v |> Exotic.extract()  # value changed if `some_c_func` modified it
```

### Transmit mechanism

If might not always be illegal memory access because the memory will be reused by the virtual machine.
The function `Exotic.Value.transmit/1` is provided to prolong the lifetime of a `Exotic.Value`
by making every new `Exotic.Value` returned by a function owns a reference to its argument `Exotic.Value`.
If you want absolute control over a `Exotic.Value` or it lives longer than any C function call,
consider using an `Agent` or a `GenServer` to store it.
If you really need some global state, you can use Erlang [`ets`](https://www.erlang.org/doc/man/ets.html) to store it.

### Transparent mode

If your array or struct is very small and will never be passed by pointer in C, you can call `Exotic.Value.transparent/1` to have Exotic use a Erlang binary to store it and make sure it is always copied.

<!-- TODO: implement this -->

### Rule of thumb

You might be confused when and whether to use transmit mechanism or transparent mode. Here are some tips:

- The data' address matters. Some functions down the line might modify/ready the data by pointer.
  Use transmit mechanism or process/ets to store the `Exotic.Value`.
- It is a small struct/array and will always be copied when passed to C function.
  Use transparent mode.
- The data is not used in any Elixir function directly, while some C function will access it.
  Must use transmit mechanism or transparent mode to avoid Erlang GC prematurely collect the data.

### Process pool for C library

Under the hood there will be a process pool for every C library instance. This design is for two concerns mainly:

- make it possible to decide if it needs to call C function on dirty scheduler (per call > 1ms)

```elixir
lib_a_instance = NativeLibA.Library.default()
# runs by a worker in the pool for calls running on normal scheduler
Exotic.call(lib_a_instance, :some_c_func, [])
NativeLibA.some_c_func()

lib_a_dirty_instance = Exotic.default(NativeLibA, dirty: :cpu)
lib_a_dirty_instance2 = Exotic.default(NativeLibA, dirty: :io)
# runs by a worker in the pool for calls running on dirty scheduler
Exotic.call(lib_a_dirty_instance, :some_c_func, [])

# instance not managed by Exotic
lib_a_dirty_instance3 = Exotic.load(NativeLibA, dirty: :io)

# to update default
:ok = Exotic.update_default(NativeLibA, dirty: :io)
```

- decouple the Erlang term encoding and decoding from the calling

```elixir
ret = NativeLibA.function_1(arg) # call C function in a process pool
some_function(ret) # caller decide what to do with the result without blocking other process using the pool
```

## Closure

You can use a erlang process as C function handler and pass it as a C function pointer.

<!-- TODO: should auto cleanup be moved to somewhere else? -->

Also you can implement a `ResourceManager` for you C value to do automatic cleanup. For instance:

```elixir
defmodule ManagedXXX do
  use ResourceManager
  @impl true
  def init(_args) do
    XXX.create()
  end
  @impl true
  def cleanup(data) do
    XXX.destroy(data)
  end
end

ManagedXXX.create()
```

## Origin

This project is highly inspired by [otter](https://github.com/cocoa-xu/otter) and it is adapted to the conventions and restrictions of Rust.

This project is originally built for an Elixir binding to LLVM/MLIR toolchain C API.
So the well-tested use cases could be limited to the style of LLVM/MLIR (which is usually C wrapper of C++).
Please send PR or issue if you find problems when you are building bindings to other C libraries.

## Major design principles and limitations

**(please check if any will break your usage of a C lib)**

- Make it easy to call basic C functions but not intended to support every possible usage.

  This means you could build a support/helper lib to wrap some complicated C/C++ code and have exotic load it.
  This also means this project is actually for those who don't want to write NIFs but don't mind writing some simple C/C++.
  If this is not enough for you, a real C/C++/Rust NIF is the way.

- Decouple type and value.

  Instead burying some conventions in NIFs, the type declarations and real argument passing are fully dynamic in Elixir.

- Bring in your own macros and functions

  If you are not satisfied with the shipped macros to declare FFIs you could also build your own macros with the exotic's NIFs which are pretty simple and straightforward.

- Type and lifetime safety at FFI calls.

  This means creating typed arguments explicitly, for example: `some_ffi(Exotic.NIF.get_c_string_value("some_binary"))`
  or`another_ffi(Exotic.get_i64_value(10))`. Basic types like integer or binary string will be automatically wrapped.

- Stateful library loading.

  <!-- TODO: support registering library loading supervised when app starts -->

  Sometimes the loading of a dynamic library is actually a part of some functionality of a C lib.
  You will keep a instance of your C lib loaded with `some_lib = Exotic.NIF.get_lib(path_to_lib)`

## About native modules

- `Exotic.NIF` provides APIs to load C libraries and call C functions.
- `Exotic.CodeGen` is used to generate Elixir code from C headers.

### Difference

`Exotic.NIF` is more performance-sensitive so some of its NIFs could be verbose and redundant. There NIFs usually have arguments of bare types like a simple rust enum or struct without complicated Erlang Term types. This is intended to minimize the overhead of Erlang Term encoding and decoding.

`Exotic.CodeGen` is less performance-sensitive so it could have complicated types to represent details of C types.
