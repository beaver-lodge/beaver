# Contributing to Beaver

This guide covers native toolchains, development, testing, and releases.
For package installation and a runnable example, start with the
[README](README.md). For the DSL, utilities, and native binding boundaries,
see [Beaver architecture](guides/architecture.md).

## Development

1. Install Elixir 1.18 or later, [see installation guide](https://elixir-lang.org/install.html)
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

Native builds require the LLVM main revision pinned in `native-deps.json`, or
a newer revision with the same APIs; older toolchains are unsupported. The
normal installer verifies upstream CI artifacts and falls back to building
the exact source revision if they are unavailable. Pass `--from-source` to
select that build explicitly. The separate CI-manifest mode described below
does not allow a source-build fallback.

- Option 1: Install the pinned `llvm/eudsl` CI artifact

  The `beaver.install_prebuilt_llvm` Mix task selects the platform ZIP pinned
  in `native-deps.json`, verifies its SHA-256 and LLVM source revision, and
  installs it into a local prefix. It lives in `scripts/install_llvm`, so
  running it never compiles Beaver itself. Explicit release asset overrides
  remain available through the standalone `mlir` package.
  For local installer development, set `MLIR_TOOLCHAIN_PATH` and
  `OPENAI_TRITON_PATH` to the absolute paths of their checkouts.

  ```bash
  bash scripts/install-prebuilt-llvm.sh priv/llvm-prebuilt
  export LLVM_CONFIG_PATH=$PWD/priv/llvm-prebuilt/bin/llvm-config
  ```

  The shell script is only a thin compatibility wrapper; the equivalent direct
  invocation is:

  ```bash
  (cd scripts/install_llvm && mix deps.get && mix beaver.install_prebuilt_llvm \
    --install-dir "$PWD/../../priv/llvm-prebuilt")
  ```

  For Triton's paired LLVM, run
  `mix openai_triton.install --llvm-only --triton-ref COMMIT` in the same project.
  To build Triton core against the selected LLVM, run `mix beaver.build_triton`
  from `scripts/install_llvm`, supplying `--source PATH`, `--llvm PATH`, and
  `--output PATH`. Use the same LLVM prefix as Beaver's native build.

  `Check Upstream` deliberately uses a separate CI channel: release tarballs
  may lag behind Beaver's required LLVM semantics. The workflow first runs
  `mix beaver.resolve_upstream_llvm`, freezes one official `llvm/eudsl` build
  from `main` or `update-llvm`, and passes the same JSON manifest to every lane
  via `--ci-manifest`. All four native LLVM build jobs must have succeeded;
  unrelated downstream jobs do not decide artifact availability. The manifest
  records the source revision, run, artifact IDs and SHA-256 digests. Missing,
  expired or invalid artifacts fail closed, without a source-build fallback.
  The native semantic probes and full tests still gate compatibility. This
  does not update production pins. Use the workflow's `llvm_build_run` input
  to replay a specific build, or `latest` to discover the newest one.

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

## Documentation examples

Mark executable Elixir fences with `<!-- beaver:test:NAME -->`.
`Beaver.Test.DocumentationExample.read!/2` selects by repository-relative path
and marker, rejecting missing/duplicate markers and misplaced fences.
Tests run the README and pass pipeline as written; IR/bytecode tests share the
control-flow guide snippet. Register guides in `mix.exs`'s `docs/0` extras.
With the native toolchain ready, run:

```bash
mix test test/documentation_example_test.exs test/documentation_examples_test.exs \
  test/gen_ir_test.exs test/bytecode_test.exs
```

## Release a new version

### Update Elixir source

- Bump versions in [`README.md`](README.md) and [`mix.exs`](mix.exs)

### Linux

- Run CI, which generates the new GitHub release in https://github.com/beaver-lodge/beaver/releases.
  Release uploads use the workflow-scoped `GITHUB_TOKEN`; no separate release token is required.
- Update release url in [`mix.exs`](mix.exs)
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
