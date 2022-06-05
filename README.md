# Beaver 🦫

Beaver, a LLVM/MLIR Toolkit in Elixir.

## Motivation

In the de-facto way of using MLIR, we need to work with C/C++, TableGen, CMake and Python (in most of cases). Each language or tool here has some functionalities and convenience we want to leverage. There is nothing wrong choosing the most popular and upstream-supported solution, but having alternative ways to build MLIR-based projects is still valuable or at least worth trying.

Elixir could actually be a good fit as a MLIR front end. Elixir has SSA, pattern-matching, pipe-operator. We can use these language features to define MLIR patterns and pass pipeline in a natural and uniformed way. Elixir is strong-typed but not static-typed which makes it a great choice for quickly building prototypes to validate and explore new ideas.

Here is a small example to showcase what it is like to define and run passes in Beaver:

```elixir
defmodule ToyPass do
  use Beaver.Pass

  pattern replace_test_op(t = %test.op{}) do
    erase(t)
    create(%test.success{})
  end

  def run(module) do
    module |> MLIR.Module.walk(patterns: [replace_test_op])
  end
end

defmodule ToyCompiler do
  import MLIR.{Passes, Sigils}
  def demo() do
    ~m"""
    module @ir {
      "test.op"() { test_attr } : () -> ()
    }
    """ |> ToyPass.run |> canonicalize |> cse |> llvm |> MLIR.ExecutionEngine.run!()
  end
end
```

Also an example to build and verify a piece of IR in Beaver:

```elixir
defmodule Toy do
  require Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.{Builtin, Func, Arith, CF}
  import Builtin, only: :macros
  import Func, only: :macros
  import MLIR, only: :macros
  import MLIR.Sigils

  def gen_some_ir() do
    Beaver.mlir do
      Builtin.module do
        Func.func some_func() do
          region do
            block bb_entry() do
              v0 = Arith.constant({:value, ~a{0: i32}}) :: ~t<i32>
              cond0 = Arith.constant(true)

              CF.cond_br(cond0, :bb1, {:bb2, [v0]})
            end

            block bb1() do
              v1 = Arith.constant({:value, ~a{0: i32}}) :: ~t<i32>
              CF.br({:bb2, [v1]})
            end

            block bb2(arg :: ~t<i32>) do
              v2 = Arith.constant({:value, ~a{0: i32}}) :: ~t<i32>
              add = Arith.addi(arg, v2) :: ~t<i32>
              Func.return(add)
            end
          end
        end
      end
    end
  end
end

Toy.gen_some_ir()
|> MLIR.Operation.dump()
|> MLIR.Operation.verify!()
```

From a high-level perspective, we are going through some interesting changes of the way we utilize accelerated-computing. The most notable project in this trend could be PyTorch. Before PyTorch, one typical way to unleash the full power of GPU to use a game engine, which is usually a library of fixed pipelines, iterations built with heavy weight languages like C++, and app developers use a more script-able language like Lua/C# to build the app/game as an extension to the engine. In ML world, TensorFlow 1 could be regarded as such as well. In contrast, the PyTorch approach puts the app developer in full command of control plane. They get to use Python to build the main loop themselves while having the full access to the accelerated-computing capability. This overturn significantly boosts the productivity and flexibility. As we can see, PyTorch already gets widespread adoption in ML and beyond.

Beaver is trying to adapt this design in Erlang/Elixir, which has great support for concurrency and fault-tolerance. Considering Elixir being more compiler-friendly as a functional programming language, we can use MLIR to build a powerful and flexible compiler stack to offload the number crunching to accelerators while still keeping the distributed and fault-tolerant Erlang vibe. Hopefully, with Beaver we could build ML, 3D and new kinds of software not possible before.

## Goals

<!-- TODO: ask Jose for advise on selling this better -->

- Powered by Elixir's composable modularity and meta-programming features, provide a simple, intuitive, and extensible interface for MLIR.
- Compile Elixir to native/WASM/GPU with the help from MLIR.
- Revisit and reincarnate symbolic AI in the HW-accelerated world. Erlang/Elixir has a [Prolog](https://en.wikipedia.org/wiki/Prolog) root!
- Introduce a new stack to machine learning.
  - Higher-level: Elixir
  - Representation: MLIR
  - Lower-level: Rust/Zig

## Why is it called Beaver?

Beaver is an umbrella species increase biodiversity. We hope this project could enable other compilers and applications in the way a beaver pond becomes the habitat of many other creatures. Many Elixir projects also use animal names as their package names and it is often about raising awareness of endangered species. To read more about why beavers are important to our planet, check out [this National Geographic article](https://www.nationalgeographic.com/animals/article/beavers-climate-change-conservation-news).

## Quick introduction

Beaver is essentially LLVM/MLIR on Erlang/Elixir. It is kind of interesting to see a crossover of two well established communities and four sub-communities. Here are some brief information about each of them.

### For Erlang/Elixir forks

- Explain this MLIR thing to me in one sentence

  MLIR is like the XML/JSON for compilers. You can build your own compiler with it or use it to "talk" to other compilers with MLIR support.

- Check out [the home page](https://mlir.llvm.org/) of MLIR.

### For LLVM/MLIR forks

- What's so good about this programming language Elixir?

  - It gets compiled to Erlang and runs on BEAM (Erlang's VM). So it has all the fault-tolerance and concurrency features of Erlang.
  - As a Lisp, Elixir has all the good stuff of a Lisp-y language including hygienic macro, protocol-based polymorphism.
  - Elixir has a powerful [higher-order module system](https://elixir-lang.org/getting-started/module-attributes.html) to persist compile-time data and this allows library users to easily adjust runtime behavior.
  <!-- TODO: some rephrase -->

- Check out [the official guide](https://elixir-lang.org/getting-started/introduction.html) of Elixir.

## Getting started

<!-- TODO: single .exs example -->

### Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `beaver` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:beaver, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/beaver>.

## How it works?

To implement a MLIR toolkit, we at least need these group of APIs:

- IR API, to create and update Ops and blocks in the IR
- Pass API, to create and run passes
- Pattern API, in which you declare the transformation of a specific structure of Ops

We implement the IR API and Pass API with the help of the [MLIR C API](https://mlir.llvm.org/docs/CAPI/). There are both lower level APIs generated from the C headers and higher level APIs that are more idiomatic in Elixir.
The Pattern API is implemented with the help from the [PDL dialect](https://mlir.llvm.org/docs/Dialects/PDLOps/). We are using the lower level IR APIs to compile your Elixir code to PDL. Another way to look at this is that Elixir/Erlang pattern matching is serving as a frontend alternative to [PDLL](https://mlir.llvm.org/docs/PDLL/).

One example:

- Elixir code:

  ```elixir
  def fold_transpose(t = %TransposeOp{a: %TransposeOp{}}) do
    replace(t, with: t.a)
  end
  ```

<!-- TODO: figure out if what we really need is pdl interp -->

- will be compiled to:

  ```mlir
  pdl.pattern : benefit(1) {
    %resultType = pdl.type
    %inputOperand = pdl.operand
    %root = pdl.operation "tosa.transpose"(%inputOperand) -> %resultType
    %val0 = pdl.result 0 of %root
    %resultType1 = pdl.type
    %inner = pdl.operation "tosa.transpose"(%val0) -> %resultType1
    pdl.rewrite %root {
      pdl.replace %root with (%inputOperand)
    }
  }
  ```

## Design principles

### Transformation over builder

It is very common to use builder pattern to construct IR, especially in a OO programming language like C++/Python.
One problem this approach has is that the compiler code looks very different from the code it is generating.
Because Erlang/Elixir is SSA by its nature, in Beaver a MLIR Op's creation is very declarative and its container will transform it with the correct contextual information. By doing this, we could:

- Keep compiler code's structure as close as possible to the generated code, with less noise and more readability.
- Allow dialects of different targets and semantic to introduce different DSL. For instance, CPU, SIMD, GPU could all have their specialized transformation tailored for their own unique concepts.

One example:

```elixir
Buildin.module do
  v2 = Arith.constant(1) :: ~t<i32>
end
# Buildin.module is a macro, it will transformed the SSA `v2= Arith.constant..` to:
v2 = Arith.constant(1, return_type: ~t<i32>, insertion_point: ..., location: ...)
```

## Is Beaver a compiler or binding to LLVM/MLIR or what?

Elixir is a programming language built for all purposes. There are multiple sub-ecosystems in the general Erlang/Elixir ecosystem.
Each sub-ecosystem appears distinct/unrelated to each other, but they actually complement each other in the real world production.
To name a few:

- [Phoenix Framework](https://phoenixframework.org/) for web application and realtime message
- [Nerves Project](https://www.nerves-project.org/) for embedded device and IoT
- [NX](https://github.com/elixir-nx/nx) for tensor and numerical

Each of these sub-ecosystems starts with a seed project/library. Beaver should evolve to become a sub-ecosystem for compilers built with Elixir and MLIR.

## Why Beaver uses C, C++ and Rust?

Although this has the downside being confusing and overwhelming, mainly there are these considerations:

- The most convenient way to ship a C/C++ library with a Elixir project is to use [the rustler project](https://github.com/rusterlium/rustler) to build a dynamic library.
- The recommended way to build a non C++ binding to MLIR is to create an "aggregate" which is an achieve of all symbols of LLVM/MLIR APIs you want to use in one shared library.
- Use libFFI to call functions dynamically whenever it is possible. Although this is less safe, it makes it possible to use macro to generate Elixir code from headers instead of writing hundreds of NIFs for all LLVM/MLIR API.

Here is the hierarchy of a typical function call in Beaver:

- Higher level API in Elixir
- libFFI in Elixir ([Exotic](/apps/exotic/README.md))
- libFFI in Rust
- default MLIR C API and [some extensions](/apps/mlir/native/mlir_nif/README.md), built as a dynamic library by Cargo and CMake
- MLIR C++ API (wrap and unwrap following MLIR C API convention)

## How Beaver works with MLIR ODS definitions?

PDL really opens a door to non C++ programming languages to build MLIR tools. Beaver will reuse PDL's implementations in LSP and C++ source codegen to generate Elixir code. The prominent part is that all ODS definitions will have their correspondent Elixir [Structs](https://elixir-lang.org/getting-started/structs.html) to be used in patterns and builders. Although this is actually a hack, it is kind of reliable considering PDL will always be part of the upstream LLVM mono-repo. We could update to its new APIs as PDL's implementation evolves. As long as it provides features like code completions and code generations, there will be some APIs in PDL's implementation we could reuse to collect and query ODS meta data.

## MLIR context management

When calling higher-level APIs, it is ideal not to have MLIR context passing around everywhere. To archive this, we borrow the practice from upstream [MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/) in LLVM repo and adapt it following Erlang/Elixir idiom. The basic idea is that all higher-level APIs are backed by Erlang [`process`](https://www.erlang.org/doc/reference_manual/processes.html) and [`ets`](https://www.erlang.org/doc/man/ets.html) to keep track of the contexts involved at different level of MLIR elements. By default it uses the global MLIR context get initialized as Erlang [application](https://www.erlang.org/doc/man/application.html)s start, or you can register a MLIR context for `self()` process.

## Development

1. Install Elixir, https://elixir-lang.org/install.html
2. Install Rust, https://rustup.rs/
3. Install LLVM/MLIR

<!-- - Option #1,  -->

- build from source https://mlir.llvm.org/getting_started/

  Recommended install commands:

  ```
  cmake -B build -S llvm -G Ninja -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DCMAKE_INSTALL_PREFIX=${HOME}/llvm-install
  cmake --build build -t install
  export LLVM_CONFIG_PATH=$HOME/llvm-install/bin/llvm-config
  ```

<!-- - Option #2 (Ubuntu), Install from LLVM apt releases, https://apt.llvm.org/ -->
<!-- - Option #3 (macOS), Install from homebrew: `brew install llvm` -->

4. Run tests

- Make sure LLVM environment variable is set properly
  ```
  echo $LLVM_CONFIG_PATH
  ```
- Run elixir tests
  ```bash
  mix test
  ```
