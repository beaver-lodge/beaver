# FizZ

FizZ (**F**oreign **i**nterface **z**eal in **Z**ig) is an Elixir package for binding a C library.
The zeal here is that an Erlang/Elixir user shouldn't be required to write a NIF to call a C function.
The approach here is highly inspired by the TableGen/.inc source code generating in LLVM.

## What FizZ does

- Make NIF a purely function dispatch. So that you can deal with the complicity in C/Zig and Elixir.
- Make it possible to pattern matching a C type in Elixir.
- Everything in Fizz is a NIF resource, including primitive types like integer and C struct. This makes it possible to pass them to C functions as pointers.
- Fizz will generate a NIF function for every C function in your wrapper header, and register every C struct as NIF resource.

## Cool features in FizZ enabled by Zig

- Packing anything into a resource

  Almost all C++/Rust implementation seems to force you to map a fixed size type to a resource type.
  In fact for same resource type, you can have Erlang allocate memory of any size.
  With Zig's comptime `sizeof` you can easily pack a list of items into an array/struct without adding any abstraction and overhead. An illustration:

  ```
    [(address to item1), item1, item2, item3, ...]
  ```

  So the memory is totally managed by Erlang, and you can use Zig's comptime feature to infer everything involved.

- Saving lib/include path to a Zig source and use them in your `build.zig`. You can use Elixir to find all the paths. It is way better than do it with make/CMake because you got a whole programming language to do it.
- Inter NIF resource exchange. Because it is Zig, just import the Zig source from another Hex package.

## Difference from Zigler

Fizz borrows a lot of good ideas ~~and code~~ from Zigler (Zigler is awesome~) but there are some differences:

- Fizz's primary goal is to help you consume a C library, not helping you write NIFs in Zig. With Fizz you don't write NIF directly, instead you provide NIF resource type conversion functions.
- Fizz expects you to have a `build.zig` to build the Zig source generated from C header along with your hand-written Zig code. So if you want to also sneak CMake inside, go for it.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `fizz` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:fizz, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/fizz>.
