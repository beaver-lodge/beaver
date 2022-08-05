# Beaver 🦫

**Boost the almighty blue-silver dragon with some magical elixir!** 🧙🧙‍♀️🧙‍♂️

## Motivation

In the de-facto way of using MLIR, we need to work with C/C++, TableGen, CMake and Python (in most of cases). Each language or tool here has some functionalities and convenience we want to leverage. There is nothing wrong choosing the most popular and upstream-supported solution, but having alternative ways to build MLIR-based projects is still valuable or at least worth trying.

Elixir could actually be a good fit as a MLIR front end. Elixir has SSA, pattern-matching, pipe-operator. We can use these language features to define MLIR patterns and pass pipeline in a natural and uniformed way. Elixir is strong-typed but not static-typed which makes it a great choice for quickly building prototypes to validate and explore new ideas.

Here is an example to build and verify a piece of IR in Beaver:

```elixir
mlir do
  module do
    Func.func some_func(function_type: Type.function([], [Type.i(32)])) do
      region do
        block bb_entry() do
          v0 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
          cond0 = Arith.constant(true) >>> Type.i(1)
          CF.cond_br(cond0, MLIR.__BLOCK__(bb1), {MLIR.__BLOCK__(bb2), [v0]}) >>> []
        end

        block bb1() do
          v1 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
          _add = Arith.addi(v0, v0) >>> Type.i(32)
          CF.br({MLIR.__BLOCK__(bb2), [v1]}) >>> []
        end

        block bb2(arg >>> Type.i(32)) do
          v2 = Arith.constant(value: Attribute.integer(Type.i(32), 0)) >>> Type.i(32)
          add = Arith.addi(arg, v2) >>> Type.i(32)
          Func.return(add) >>> []
        end
      end
    end
    |> MLIR.Operation.verify!(dump_if_fail: true)
  end
end
|> MLIR.Operation.verify!(dump_if_fail: true)
```

And a small example to showcase what it is like to define and run a pass in Beaver (with some monad magic):

```elixir
alias Beaver.MLIR.Dialect.Func

defmodule ToyPass do
  use Beaver.MLIR.Pass, on: Func.Func

  defpat replace_add_op(_t = %TOSA.Add{operands: [a, b], results: [res], attributes: []}) do
    %TOSA.Sub{operands: [a, b]}
  end

  def run(%MLIR.CAPI.MlirOperation{} = operation) do
    with %Func.Func{attributes: attributes} <- Beaver.concrete(operation),
          2 <- Enum.count(attributes),
          {:ok, _} <- MLIR.Pattern.apply_(operation, [replace_add_op()]) do
      :ok
    end
  end
end

~m"""
module {
  func.func @tosa_add(%arg0: tensor<1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
"""
|> MLIR.Pass.Composer.nested(Func.Func, [
  ToyPass.create()
])
|> canonicalize
|> MLIR.Pass.Composer.run!()
```

From a high-level perspective, we are going through some interesting changes of the way we utilize accelerated-computing. The most notable project in this trend could be PyTorch. Before PyTorch, one typical way to unleash the full power of GPU is to use a game engine, usually in the form of a library of fixed pipelines, iterations built with heavy weight languages like C++, and app developers use a more script-able language like Lua/C# to build the app/game as an extension to the engine. In ML world, TensorFlow 1 could be regarded as such as well. In contrast, the PyTorch approach puts the app developer in full command of control plane. They get to use Python to build the main loop themselves while having the full access to the accelerated-computing capability. This overturn significantly boosts the productivity and flexibility. As we can see, PyTorch already gets widespread adoption in ML and beyond.

Beaver is trying to adapt this design in Erlang/Elixir, which has great support for concurrency and fault-tolerance. Considering Elixir being more compiler-friendly as a functional programming language, we can use MLIR to build a powerful and flexible compiler stack to offload the number crunching to accelerators while still keeping the distributed and fault-tolerant Erlang vibe. Hopefully, with Beaver we could build ML, 3D and new kinds of software not possible before.

## Goals

<!-- TODO: ask Jose for advise on selling this better -->

- Powered by Elixir's composable modularity and meta-programming features, provide a simple, intuitive, and extensible interface for MLIR.
- Edit-Build-Test-Debug Loop at seconds. Everything in Elixir and Zig are compiled in parallel.
- Compile Elixir to native/WASM/GPU with the help from MLIR.
- Revisit and reincarnate symbolic AI in the HW-accelerated world. Erlang/Elixir has [a Prolog root](https://www.erlang.org/blog/beam-compiler-history/#the-prolog-interpreter)!
- Introduce a new stack to machine learning.
  - Higher-level: Elixir
  - Representation: MLIR
  - Lower-level: Zig

## Why is it called Beaver?

If it has to be an abbreviation. It could be **BEA**M **Ve**rsatile **R**epresentation. Beaver is an umbrella species increase biodiversity. We hope this project could enable other compilers and applications in the way a beaver pond becomes the habitat of many other creatures. Many Elixir projects also use animal names as their package names and it is often about raising awareness of endangered species. To read more about why beavers are important to our planet, check out [this National Geographic article](https://www.nationalgeographic.com/animals/article/beavers-climate-change-conservation-news).

## Quick introduction

Beaver is essentially LLVM/MLIR on Erlang/Elixir. It is kind of interesting to see a crossover of two well established communities and four sub-communities. Here are some brief information about each of them.

### For Erlang/Elixir forks

- Explain this MLIR thing to me in one sentence

  MLIR is like the HTTP for compilers. You can build your own compiler with it or use it to "talk" to other compilers with MLIR support.

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

## Erlang apps under the Beaver umbrella project

LLVM/MLIR is a giant project, and built around that Beaver have hundreds of Erlang modules and thousands of functions. To properly ship LLVM/MLIR and streamline the development process, we need to carefully break the functionalities at different level into different Erlang apps under the same umbrella.

- `:manx`: Pure Elixir, compiler backend for [Nx](https://github.com/elixir-nx/nx/tree/main/nx#readme).
- `:beaver`: Elixir and C/C++ hybrid.
  - Top level app ships the high level functionalities including IR generation and pattern definition.
  - MLIR CAPI wrappers built by parsing LLVM/MLIR CAPI C headers and some middle level helper functions to hide the C pointer related operations. This app will add the loaded MLIR C library and managed MLIR context to Erlang supervisor tree. Rust is also used in this app, but mainly for LLVM/MLIR CMake integration.
  - All the Ops defined in stock MLIR dialects, built by querying the registry. This app will ship MLIR Ops with Erlang idiomatic practices like behavior compliance.
- `:kinda`: Elixir and Zig hybrid, generating NIFs from MLIR C headers.

### Notes on consuming and development

- Only `:beaver` and `:kinda` are designed to be used as stand-alone app being directly consumed by other apps.
- `:manx` could only work with Nx.
- Although `:kinda` is built for Beaver, any Erlang/Elixir app with interest bundling some C API could take advantage of it as well.
- The namespace `Beaver.MLIR` is for standard features are generally expected in any MLIR tools.
- The namespace `Beaver` is for concepts and practice only exists in Beaver, which are mostly in a DSL provided as a set of macros (including `mlir/0`, `block/1`, `defpat/2`, etc). The implementations are usually under `Beaver.DSL` namespace.
- In Beaver, there is no strict requirements on the consistency between the Erlang app name and Elixir module name. Two modules with same namespace prefix could locate in different Erlang apps (this happens a lot to the `Beaver.MLIR` namespace). Of course redefinition of Elixir modules with an identical name should be avoided.

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

It is very common to use builder pattern to construct IR, especially in an OO programming language like C++/Python.
One problem this approach has is that the compiler code looks very different from the code it is generating.
Because Erlang/Elixir is SSA by its nature, in Beaver a MLIR Op's creation is very declarative and its container will transform it with the correct contextual information. By doing this, we could:

- Keep compiler code's structure as close as possible to the generated code, with less noise and more readability.
- Allow dialects of different targets and semantic to introduce different DSL. For instance, CPU, SIMD, GPU could all have their specialized transformation tailored for their own unique concepts.

One example:

```elixir
Buildin.module do
  v2 = Arith.constant(1) >>> ~t<i32>
end
# Buildin.module is a macro, it will transformed the SSA `v2= Arith.constant..` to:
v2 = Arith.constant(value: ~a{1}, return_type: ~t<i32>)
```

Also, using the declarative way to construct IR, proper dominance and operand reference is formed naturally.

<!-- TODO: use real code here -->

```elixir
SomeDialect.some_op do
  region do
    block entry() do
      x = Arith.constant(1) >>> ~t<i32>
      y = Arith.constant(1) >>> ~t<i32>
    end
  end
  region do
    block entry() do
      z = Arith.addi(x, y) >>> ~t<i32>
    end
  end
end

# will be transformed to:

SomeDialect.some_op(
  regions: fn -> do
    region = MLIR.__REGION__() # first region created
    block = MLIR.__BLOCK__()
    x = Arith.constant(...)
    y = Arith.constant(...)

    region = MLIR.__REGION__() # second region created
    block = MLIR.__BLOCK__()
    z = Arith.addi([x, y, ...]) # x and y dominate z
  end
)
```

### Beaver DSL as higher level AST for MLIR

There should be a 1:1 mapping between Beaver SSA DSL to MLIR SSA. It is possible to do a roundtrip parsing MLIR text format and dump it to Beaver DSL which is Elixir AST essentially. This makes it possible to easily debug a piece of IR in a more programmable and readable way.

In Beaver, working with MLIR should be in one format, no matter it is generating, transforming, debugging.

### High level API in Erlang/Elixir idiom

- When possible, lower level C APIs should be wrapped as Elixir struct with support to common Elixir protocols.
  For instance the iteration over one MLIR operation's operands, results, successors, attributes, regions should be implemented in Elixir's Enumerable protocol.
  This enable the possibility to use the rich collection of functions in Elixir standard libraries and Hex packages.
- Erlang Modules to work with Ops in different MLIR dialects should implement behaviors like `Beaver.DSL.Op.Prototype`.

## Is Beaver a compiler or binding to LLVM/MLIR?

Elixir is a programming language built for all purposes. There are multiple sub-ecosystems in the general Erlang/Elixir ecosystem.
Each sub-ecosystem appears distinct/unrelated to each other, but they actually complement each other in the real world production.
To name a few:

- [Phoenix Framework](https://phoenixframework.org/) for web application and realtime message
- [Nerves Project](https://www.nerves-project.org/) for embedded device and IoT
- [Nx](https://github.com/elixir-nx/nx) for tensor and numerical

Each of these sub-ecosystems starts with a seed project/library. Beaver should evolve to become a sub-ecosystem for compilers built with Elixir and MLIR.

## How Beaver works with MLIR ODS definitions?

PDL really opens a door to non C++ programming languages to build MLIR tools. Beaver will reuse PDL's implementations in LSP and C++ source codegen to generate Elixir code. The prominent part is that all ODS definitions will have their correspondent Elixir [Structs](https://elixir-lang.org/getting-started/structs.html) to be used in patterns and builders. Although this is actually a hack, it is kind of reliable considering PDL will always be part of the upstream LLVM mono-repo. We could update to its new APIs as PDL's implementation evolves. As long as it provides features like code completions and code generations, there will be some APIs in PDL's implementation we could reuse to collect and query ODS meta data.

## MLIR context management

When calling higher-level APIs, it is ideal not to have MLIR context passing around everywhere. To achieve this, we borrow the practice from upstream [MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/) in LLVM repo and adapt it following Erlang/Elixir idiom. The basic idea is that all higher-level APIs are backed by Erlang [`process`](https://www.erlang.org/doc/reference_manual/processes.html) and [`ets`](https://www.erlang.org/doc/man/ets.html) to keep track of the contexts involved at different level of MLIR elements. By default it uses the global MLIR context get initialized as Erlang [application](https://www.erlang.org/doc/man/application.html)s start, or you can register a MLIR context for `self()` process.

## Development

1. Install Elixir, https://elixir-lang.org/install.html
2. Install Zig, https://ziglang.org/learn/getting-started/#installing-zig
3. Install LLVM/MLIR

- build from source https://mlir.llvm.org/getting_started/

  Recommended install commands:

  ```
  cmake -B build -S llvm -G Ninja -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=${HOME}/llvm-install
  cmake --build build -t install
  export LLVM_CONFIG_PATH=$HOME/llvm-install/bin/llvm-config
  ```

  To use Vulkan:

  - Install Vulkan SDK (global installation is required), reference: https://vulkan.lunarg.com/sdk/home
  - Setting environment variable by adding commands these to your bash/zsh profile:

    ```
    # you might need to change the version here
    cd $HOME/VulkanSDK/1.3.216.0/
    source setup-env.sh
    cd -
    ```

  - use `vulkaninfo` and `vkvia` to verify Vulkan is working
  - Add `-DMLIR_ENABLE_VULKAN_RUNNER=ON` in LLVM CMake config command

4. Run tests

- Clone the repo
- Make sure LLVM environment variable is set properly, otherwise it might fail to build
  ```bash
  echo $LLVM_CONFIG_PATH
  ```
- Build and run Elixir tests
  ```bash
  mix deps.get
  mix test
  # run tests with filters
  mix test --exclude vulkan # use this to skip vulkan tests
  mix test --only smoke
  mix test --only nx
  ```

5. debug

- setting environment variable to control Erlang scheduler number, `ERL_AFLAGS="+S 10:5"`
- run mix test under LLDB, `scripts/lldb-mix-test`

6. Livebook

- To use Beaver in [Livebook](https://livebook.dev/), run this in the source directory:
  ```bash
  livebook server --name livebook@127.0.0.1 --home . --default-runtime mix
  ```
