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

Bindings are the part that provides the interface to the MLIR CAPIs. It is implemented in Zig and is responsible for calling MLIR functions. Note that Beaver's bindings will try not to use `TableGen` and instead try to make use Elixir and Zig's meta-programming features to generate the bindings.

## Development

1. Install Elixir, [see installation guide](https://elixir-lang.org/install.html)
2. Install Zig, [see installation guide](https://ziglang.org/learn/getting-started/#installing-zig)
3. Install LLVM/MLIR

- Option 1: Install with pip

  ```bash
  python3 -m pip install -r dev-requirements.txt
  export LLVM_CONFIG_PATH=$(python3 -c 'import mlir;print(mlir.__path__[0])')/bin/llvm-config
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

4. Develop and run tests
- Clone this repo and `kinda` in the same directory
  ```bash
  git clone https://github.com/beaver-lodge/beaver.git
  git clone https://github.com/beaver-lodge/kinda.git
  ```
- Make sure LLVM environment variable is set properly, as otherwise it might fail to build

  ```bash
  echo $LLVM_CONFIG_PATH
  ```

- Build and run Elixir tests
  ```bash
  mix deps.get
  BEAVER_BUILD_CMAKE=1 mix test
  # run tests with filters
  mix test --exclude vulkan # use this to skip vulkan tests
  mix test --only smoke
  mix test --only nx
  ```

5. debug

- setting environment variable to control Erlang scheduler number, `ERL_AFLAGS="+S 10:5"`
- run mix test under LLDB, `scripts/lldb-mix-test`

## Release a new version

### Update Elixir source

- Bump versions in [`README.md`](README.md) and [`mix.exs`](/mix.exs)

### Linux

- Run CI, which generates the new GitHub release uploaded to https://github.com/beaver-lodge/beaver-prebuilt/releases.
- Update release url in [`mix.exs`](/mix.exs)

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
BEAVER_BUILD_CMAKE=1 mix hex.publish
```

## Format CMake files

```bash
python3 -m pip install cmake-format
cmake-format -i native/**/CMakeLists.txt native/**/*.cmake
```

## Erlang apps in Beaver

LLVM/MLIR is a giant project, and built around that Beaver have thousands of functions. To properly ship LLVM/MLIR and streamline the development process, we need to carefully break the functionalities at different level into different Erlang apps under the same umbrella.

- `:beaver`: Elixir and C/C++ hybrid.
  - Top-level app ships the high-level functionalities including IR generation and pattern definition.
  - MLIR CAPI wrappers built by parsing LLVM/MLIR CAPI C headers and some middle level helper functions to hide the C pointer related operations. This app will add the loaded MLIR C library and managed MLIR context to Erlang supervisor tree. Rust is also used in this app, but mainly for LLVM/MLIR CMake integration.
  - All the Ops defined in stock MLIR dialects, built by querying the registry. This app will ship MLIR Ops with Erlang idiomatic practices like behavior compliance.
- `:kinda`: Elixir and Zig hybrid, generating NIFs from MLIR C headers. Repo: https://github.com/beaver-lodge/kinda

## Miscellaneous

Some other notes on consuming and development

- Only `:beaver` and `:kinda` are designed to be used as stand-alone app being directly consumed by other apps.
- `:manx` could only work with Nx.
- Although `:kinda` is built for Beaver, any Erlang/Elixir app with interest bundling some C API could take advantage of it as well.
