# Introducing Beaver 🦫

[Beaver](https://github.com/beaver-project/beaver) is a toolkit to build MLIR-based compilers with Elixir.
By distilling all the amazing features of MLIR in idiomatic Erlang/Elixir, the functional programming in it makes the "progressive lowering" philosophy of MLIR shine!

## Beaver's AI mission

Being this special mixture of Erlang/Elixir and LLVM/MLIR, Beaver should enable some amazing innovations in the machine learning world.

### Neural network looks like a network

IMO, lots of figures deposit the neural network algorithm in a pretty deceptive way by portraying it as a network. It is not the case. Information flow in an artificial neural network (ANN) is very directive. It doesn't have the interconnection a real biology neural network (BNN) has among its neurons.

It's been ten years since the AlexNet breakthrough, maybe we should start building something dramatically different from ANN and looks like BNN. The programming language and paradigm widely used in ML could be limiting us here. They are sequential and imperative. They are not built with concurrency in mind, like how a real network works: the information flows in all directions and gets exchanged freely at an ultra-high frequency.

In contrast, being invented by a physicist, Erlang was built to represent and emulate the concurrency nature of real-world physical laws. Even running on a single node, Erlang can easily keep tens of thousands of connections simultaneously. How about letting it run a neural network processing data from 100 cameras on a self-driving car?

### Built for software 2.0

As a programming language with strong meta-programming capability, Elixir itself and many popular Elixir libraries are built with meta-programming. So it is just natural to insert some AI-generated code into an Elixir AST as if it is inserted by a macro written by a human.

We could introduce different AI-guided transformations on different levels of code/IR including Elixir AST or MLIR and reuse all the existing checks and verifications in the Elixir compiler and MLIR dialects. What's even better, all these codes can be fed to LLVM and generate the code that can run on any CPU/GPU/accelerator.

### Prolog root and symbolic AI

Erlang/Elixir could be the only mainstream/kind of popular programming language that still keeps the symbolic AI vibe that Prolog once pioneered.
After 30 years of evolvement, there are already well-established design patterns, and best practices in Erlang/Elixir community to write software in this Prolog-clause style.
What Beaver does is to to do is to extend this clause-based programming style by compiling some Elixir code to tensor representation and using MLIR to generate CPU/GPU code. so that human-written and readable code can work with neural networks of tensors.

### Fault-tolerance and distributed systems

As the size of parameters of SOTA models grows to billions and trillions, it is quite scary that there is almost no serious fault tolerance in ML.
With Erlang, we could have runtime fault-tolerance by applying the "let it crash" philosophy. This means that the AI system we are going to build with Beaver will allow errors during training or inference.
What is more exciting is that the error message of a failure can get propagated into the learning itself.

As for Erlang itself, what's cool is that now we can add a compiler for Erlang distribution. We can represent the distribution and communications with MLIR dialects like [OpenMP](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/) and optimize the overhead and latency of a distributed system. After that optimization, we can also offload some of the communication to native code generated or RDMA API calls.

## What is MLIR

[MLIR](https://mlir.llvm.org/) is a super ambitious project. It is so flexible and powerful that it compiles anything that could be represented as a graph to anything that could be run on a computer, or directly to hardware via [⚡️CIRCT](https://github.com/llvm/circt). There are even researchers using it to build a compiler for quantum computers.

## Why Elixir and MLIR are perfect for each other

Just by looking at each, there is a perfect match in terms of expression and code structure.

<table>

  <tr>
    <th>Elixir</th>
    <th>MLIR</th>
  </tr>

<tr>
<td>

<pre lang='elixir'>


ft = Type.function([Type.i32()], [Type.i32()])
i1024 = Attribute.integer(Type.i32(), 1024)
module do
  Func.func some_func(function_type: ft) do
    region do
      block bb_entry(a >>> Type.i32()) do
        b = Arith.constant(value: i1024) >>> Type.i32()
        c = Arith.addi(a, b) >>> Type.i32()
        Func.return(c) >>> []
      end
    end
  end
end


</pre>

</td>

<td>

<pre lang='mlir'>

module {
  func.func @some_func(%arg0: i32) -> i32 {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.addi %arg0, %c1024_i32 : i32
    return %0 : i32
  }
}

</pre>

</td>
</tr>
</table>

- Compiled to Erlang, everything is a [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form) in Elixir.
- Everything in MLIR is SSA.
- This resemblance of structure makes it possible to create a perfect match of the semantics in an MLIR dialect.
- Elixir has a stable AST and powerful built-in facilities to do AST transformation. When using Beaver to generate MLIR in Elixir, is essentially generating MLIR API calls from an Elixir AST, the semantics and dominance are preserved perfectly.

### Build MLIR systems in Elixir with infinite extensibility, not just one MLIR toy.

By implementing MLIR dialect or a collection of passes packaged as an Erlang OTP application and setup the supervisor tree, you can easily extend the stock MLIR features shipped by Beaver.
You don't need to build LLVM/MLIR from source code yourself, and all the pattern matching and pass orchestration will just work.
What is even crazier is that you can even use Erlang's facilities to hot-upgrade your MLIR compiler or build a distributed one if you want to.

### Profoundly, they share common philosophies and practices.

- One of the core ideas of MLIR is something called "progressive lowering". It means [......]
  Every Erlang/Elixir program is essentially a progressive lowering. It use patterns to transform data at different abstraction levels. Although there is no such formal name, almost all books about Erlang/Elixir would talk about this and roughly describe it as "think transformation". The transformation done by Elixir's `Macro` module and `Beaver.Walker` has a striking resemblance.
- Erlang/Elixir has the pervasive usage of pattern matching, even in function arguments. And pattern matching is exactly what MLIR passes are made of. So every Erlang/Elixir developer can become an LLVM/MLIR developer from day one because they have been writing SSA and pattern matching since the 1980s...

## Core features of Beaver

- Declarative MLIR generating
  - Contextual information injection without stateful insert points
- MLIR transformation in classical functional AST traversal style
- Generating PDL IR from Elixir source

  ```elixir
  defpat replace_add_op(_t = %TOSA.Add{operands: [a, b], results: [res], attributes: []}) do
    %TOSA.Sub{operands: [a, b]}
  end
  ```

- Dialect-aware monadic pattern matching

  ```elixir
  with %Func.Func{attributes: attributes} <- Beaver.concrete(operation),
        2 <- Enum.count(attributes),
        {:ok, _} <- MLIR.Pattern.apply_(operation, [replace_add_op()]) do
    :ok
  end
  ```

### To summarize the rationale:

`An MLIR compiler`

= `MLIR generation + MLIR passes`

= `MLIR SSA generation + MLIR pattern matching`

= `Elixir AST generation + Elixir pattern matching`

=~ `Everyday Erlang/Elixir code`

So **everyone** knows Erlang/Elixir could build their compiler in MLIR!

### Target **anything**!

We can write Elixir code that compiles Elixir to **anything**, because LLVM/MLIR already does so. To summarize:

`Beaver API calls` -> `Elixir AST` -> `MLIR API calls` -> `MLIR` -> `LLVM/SPIR-V/CIRCT` -> `Instructions/HW`

## Why machine learning in Elixir is a good idea

A typical Python-based machine learning library/framework consists of:

- Python (30%)
- A lot of C++/CUDA (60%)
- MLIR/LLVM in C++ (10%)

Let's build another stack

- MLIR/LLVM in Elixir (95%, with great flexibility and expressiveness)
- Zig (5%, well-defined and optimized to bits)

Machine learning is a kind of application that prioritizes throughput. The absolute single-threaded performance is not the key factor here since the number crushing is offloaded to the accelerators like GPU or SIMD instructions. What matters is that the higher-level programming language should have extremely low latency to async events of the accelerator so that the accelerator is never under-used. Erlang is just the perfect platform for this. It has second-to-none preemptive scheduling making sure all the events in the system are never delayed at the best effort. It is called "soft-realtime" in Erlang's words. On the other hand, machine learning developers always want the higher-level APIs to be flexible and expressive, even willing to pay the price of lower single-thread performance. Just look at how popular PyTorch is.

In a word, an ideal higher-level programming language for ML should be responsive (low latency to HW events) and expressive (with pattern matching, can build both model and compiler). And Erlang/Elixir happens to have it all. What is even more exciting is that all the good stuff about fault tolerance and meta-programming is yet to be popularized in the machine learning world.

### No boundary Runtime and compile time

- With Beaver, we can run a model when compiling a model, and run a compiler inside a model
- Tens of thousands of Erlang processes could be tens of thousands of different compilers generating code of their interests. They could talk to each other with tensor, IR, logic symbols, and raw data.

### Pass generated

Here is a very common use case in tensor compilers: to maximize the FLOPS of GPU and other accelerators, we often hard-coded passes to do tiling and similar optimization for memory hierarchy. With Beaver, at compile time we can generate a pass by running a benchmark to search for the best tiling parameter and persistent the configs.

## Built for purposes and applications outside of machine learning.

Not just for machine learning, but other purposes.

- MLIR gets lowered to LLVM or SPIR-V
- LLVM targets WASM
- Someday maybe we could compile the whole Erlang virtual machine to hardware. Who knows.

### Compiler as service

Remote code execution shouldn't be a bug but a feature. Instead of keeping adding commands until it is out of control, how about using IR as the "hypertext"?
