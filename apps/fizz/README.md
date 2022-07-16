# FizZ

FizZ (**F**oreign **i**nterface **z**eal in **Z**ig) is an Elixir package for binding a C library.
The zeal here is that an Erlang/Elixir user shouldn't be required to write a NIF to call a C function.
The approach here is highly inspired by the rocket-science-advanced TableGen/.inc technology in LLVM.

## What FizZ does

- Make NIF a purely function dispatch. So that you can deal with the complicity in C/Zig and Elixir.
- Make it possible to pattern matching a C type in Elixir.
- Everything in Fizz is a NIF resource, including primitive types like integer and C struct. This makes it possible to pass them to C functions as pointers.
- Fizz will generate a NIF function for every C function in your wrapper header, and register every C struct as NIF resource.

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
