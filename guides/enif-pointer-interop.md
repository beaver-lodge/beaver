# ENIF pointer interoperability

Beaver uses MLIR's upstream Ptr dialect as the IR-level pointer abstraction.
This keeps pointer intent visible until the LLVM boundary without treating a
BEAM resource as if it were an SSA value.

## Three representations

The interop boundary has three distinct representations:

1. A `Beaver.Native.OpaquePtr` or another `Beaver.Native` resource is a
   host-side owner or handle. It is managed by the BEAM/NIF runtime and does
   not belong in MLIR IR.
2. A `!ptr.ptr` value is a native pointer inside MLIR. ENIF pointer signatures
   use `!ptr.ptr<#ptr.generic_space>` so `convert-to-llvm` can map the generic
   ABI pointer to LLVM address space 0.
3. A memref carries pointer metadata and ownership-independent shape
   information. It should stay a memref until an external function actually
   requires a C pointer.

These representations must cross explicit operations or runtime boundaries;
they are not interchangeable casts.

## ENIF out-parameters

Allocate typed storage as a memref, then use `ptr.to_ptr` at the call boundary:

```elixir
alias Beaver.MLIR.Dialect.{MemRef, Ptr}

generic_space = Ptr.generic_space()
ptr_type = Ptr.type()

storage =
  MemRef.alloca(operand_segment_sizes: :infer) >>>
    Type.memref!([], Type.i64(ctx: ctx), memory_space: generic_space)

argument = Ptr.to_ptr(storage) >>> ptr_type
ENIF.get_int64(env, term, argument) >>> :infer
value = MemRef.load(storage) >>> Type.i64()
```

The memref owns the storage semantics. `ptr.to_ptr` exposes only its aligned
data pointer to the ENIF call. Do not reconstruct a memref from a raw ENIF
pointer unless the pointer's metadata, lifetime, and ownership contract are
available explicitly.

## Null and raw addresses

Use typed Ptr attributes rather than routing through an LLVM zero and an
`unrealized_conversion_cast`:

```elixir
ptr_type = Ptr.type()
null = Ptr.constant(value: Ptr.null(type: ptr_type)) >>> ptr_type
address = Ptr.constant(value: Ptr.address(0x1000, type: ptr_type)) >>> ptr_type
```

An address constant does not acquire ownership or extend a lifetime. It is
valid only when the native side guarantees that the address is meaningful for
the complete use interval.

## Loads, stores, arithmetic, and address spaces

For values that are already native pointers, use upstream `ptr.load`,
`ptr.store`, `ptr.ptr_add`, and `ptr.ptr_diff`. Vector pointer values can use
the upstream masked load/store and gather/scatter operations.

When Ptr operations remain inside an `llvm.func` until LLVM IR translation,
make the target address space explicit:

```elixir
generic_llvm_pointer = Ptr.type(memory_space: {:llvm, 0})
shared_llvm_pointer = Ptr.type(memory_space: {:llvm, 3})
```

This distinction makes lowering inspectable:

```text
ENIF/memref ABI path:
  memref<..., #ptr.generic_space>
    -> ptr.to_ptr
    -> !ptr.ptr<#ptr.generic_space>
    -> convert-to-llvm
    -> !llvm.ptr                         (address space 0)

Direct LLVM translation path:
  !ptr.ptr<#llvm.address_space<N>>
    -> MLIR LLVM translation
    -> LLVM ptr addrspace(N)
```

Resource handles and `ERL_NIF_TERM` values are not raw addresses. Keep them in
their declared integer/resource representation and cross into native pointer
IR only through an API whose ownership and lifetime contract is explicit.
