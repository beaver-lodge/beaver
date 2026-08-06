# Contributing to Beaver
This document describes how to contribute to Beaver by introducing the idea behind how Beaver group its functionalities and how to set up the development environment.

## Beaver's functionalities

There are three main parts in Beaver. Here is a brief introduction to each part:
- DSL: The syntax of Beaver SSA expression.
- Utilities: Work with MLIR in Elixir way.
- Bindings: Thin wrapper of MLIR CAPI.

### DSL
Modules including `Beaver`, `Beaver.Env`.

DSL is the core part of Beaver. It uses Elixir's syntax to express MLIR semantics.

### Utilities
Modules including `Beaver.Walker`, `Beaver.Composer`

Utilities are the helper functions that help to generate or manipulate MLIR IR. They are implemented in Elixir and is designed to be used in the DSL part to further enhance it and improve ergonomics.

### Bindings
Modules including `Beaver.MLIR`, `Beaver.MLIR.Dialect`, `Beaver.MLIR.Pass`, `Beaver.MLIR.Transform`, `Beaver.MLIR.ExecutionEngine`

Bindings are the part that provides the interface to the MLIR CAPIs. They are implemented in Zig and are responsible for calling MLIR functions. Beaver uses Zig comptime reflection over Build System C translation for native registration and a machine-readable declaration manifest for the Elixir surface; it does not generate wrapper source files.

## Development

1. Install Elixir, [see installation guide](https://elixir-lang.org/install.html)
2. Install Zig, [see installation guide](https://ziglang.org/learn/getting-started/#installing-zig)
3. Clone this repo. To iterate on Beaver and Kinda together, clone both and
   point Beaver at the Kinda checkout explicitly:

```bash
git clone https://github.com/beaver-lodge/beaver.git
git clone https://github.com/beaver-lodge/kinda.git
cd beaver
export BEAVER_KINDA_PATH=../kinda
```

Without `BEAVER_KINDA_PATH`, Mix uses the released Kinda version from
`mix.lock`. During native builds, Beaver passes that resolved Mix dependency to
Zig as a local package override, so both dependency graphs use the same source.

4. Install LLVM/MLIR

- Option 1: Install a prebuilt `llvm/eudsl` tarball

  The `beaver.install_prebuilt_llvm` Mix task selects the matching
  `mlir_<os>_<arch>_*.tar.gz` asset for the current machine, downloads it, and
  extracts the compiled binaries into a local prefix. It lives in
  `scripts/install_llvm`, a tiny standalone Mix project, so running it never
  compiles Beaver itself.

  ```bash
  bash scripts/install-prebuilt-llvm.sh priv/llvm-prebuilt
  export LLVM_CONFIG_PATH=$PWD/priv/llvm-prebuilt/bin/llvm-config
  ```

  The shell script is only a thin compatibility wrapper; the equivalent direct
  invocation is:

  ```bash
  (cd scripts/install_llvm && mix beaver.install_prebuilt_llvm \
    --install-dir "$PWD/../../priv/llvm-prebuilt")
  ```

- Option 2: Build from source https://mlir.llvm.org/getting_started/
  Recommended install commands:

  ```bash
  cmake -B build -S llvm -G Ninja -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="-fuse-ld=lld" \
    -DCMAKE_INSTALL_PREFIX=${HOME}/llvm-install
  cmake --build build -t install
  export LLVM_CONFIG_PATH=$HOME/llvm-install/bin/llvm-config
  ```

  (Optional) To use Vulkan:

  - Install Vulkan SDK (global installation is required), reference: https://vulkan.lunarg.com/sdk/home
  - Setting environment variable by adding commands these to your bash/zsh profile:

    ```bash
    # you might need to change the version here
    cd $HOME/VulkanSDK/1.3.216.0/
    source setup-env.sh
    cd -
    ```

  - Use `vulkaninfo` and `vkvia` to verify Vulkan is working
  - Add `-DMLIR_ENABLE_VULKAN_RUNNER=ON` in LLVM CMake config command

5. Develop and run tests
- Make sure LLVM environment variable is set properly, as otherwise it might fail to build

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

6. debug

- setting environment variable to control Erlang scheduler number, `ERL_AFLAGS="+S 10:5"`
- run mix test under LLDB, `scripts/lldb-mix-test`

## Release a new version

### Update Elixir source

- Bump versions in [`README.md`](README.md) and [`mix.exs`](/mix.exs)

### Linux

- Run CI, which generates the new GitHub release in https://github.com/beaver-lodge/beaver/releases.
  Release uploads use the workflow-scoped `GITHUB_TOKEN`; no separate release token is required.
- Update release url in [`mix.exs`](/mix.exs)
- Run docker image to build for ARM:
  ```bash
  docker run -it --rm -v $PWD/..:/src -w /src/beaver --env MIX_BUILD_ROOT='_build/arm' jackalcooper/beaver-livebook-arm64:latest bash scripts/build-for-publish.sh
  ```

### Mac

- Run macOS build with:

  ```bash
  rm -rf _build/prod
  bash scripts/build-for-publish.sh
  ```

- Upload the `beaver-nif-[xxx].tar.gz` file to release

### Generate `checksum.exs`

```
rm checksum.exs
mix clean
mix
mix elixir_make.checksum --all --ignore-unavailable --print
```

Check the version in the output is correct.

### Publish to Hex

```
mix hex.publish
```

## Format CMake files

```bash
python3 -m pip install cmake-format
cmake-format -i native/**/CMakeLists.txt native/**/*.cmake
```
