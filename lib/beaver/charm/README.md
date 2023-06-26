# Charm

(Mojo)[https://www.modular.com/mojo] has realized many of the things that Beaver was originally built for, like compiling to MLIR and embedding MLIR pieces (aka, staging).
Following the evolvement of Mojo (here)[https://docs.modular.com/mojo/changelog.html#september-2022] the `Beaver.Charm` module will provide you with functions to compile Charm-compatible Elixir code to MLIR and eventually machine code.

## Why do we need it?

This question could be restated as "What is the purpose of low level programming in Erlang and Elixir?"
Previously, Erlang were primarily utilized for higher-level programming.
The [Chisel](https://www.chisel-lang.org) project.
the correctness and conciseness of FP.
The widely popular ML library PyTorch and TF demonstrate the effectiveness of a computation graph API could be.

- When it comes to "accelerated compute", programming languages ought to have the ability to target hardware. Typically, a C/C++ library is enclosed in a Python package as the prevailing method. However, we feel that a more encouraging approach exists, which will be explored further in the section titled [How is Mojo/Charm's approach different from other ML libraries?].
- An accelerator has many "magical number". One way you can hardcoded optimization for them (different cache sizes, etc),
  or the programming language could ship an auto-tuner to automate the parameter search and according code generation.

## Road map

- Lexer and parser for parametric function
- Parametric result, compile-time constant
- Reference argument
- Magical functions of struct
- Auto-tuner API

## Ideas

- Reuse the Elixir environment as the compile-time for Charm
- Implement magical methods for different arithmetical operations as callbacks for behavior
- Ownership with message passing as boundary. Given the actor model being pervasive in Erlang,
  Charm should introduce a ownership reflects and optimized for the actor model.
  Values used by current process can be mapped to reused address and register while
  one might choose to transfer the ownership of a large blob when sending the message to another process.
- `defstage` macro defines MLIR generation and code for Elixir compiler (usually a noop)

  ```elixir
  defmodule Beaver.Charm.Lifetime do
    use Beaver.Charm.Stage
    defstage move(value) do
      Beaver.Charm.Dialect.Lifetime.move(value) >>> MLIR.CAPI.mlirValueGetType(value)
    elixir # this could be omitted when the arity is 1
      value
    end
  end
  ```

  - When charm compiler is calling `Kernel.ParallelCompiler`,
    it will extract the MLIR generation code from BEAM SSA and eval it in a managed environment.

## Magical methods

A straightforward solution is to mimic what Mojo is doing with [PythonObject](https://docs.modular.com/mojo/MojoPython/PythonObject.html).
The problem in Erlang's world is that it is not object-oriented. There are no such fixed number of magical methods to implement.

## References

- [Boolean example](https://docs.modular.com/mojo/notebooks/BoolMLIR.html) in mojo
