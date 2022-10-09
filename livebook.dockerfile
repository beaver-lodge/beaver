FROM livebook/livebook as base

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y \
    build-essential git unzip wget ninja-build ca-certificates cmake ccache && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# setup llvm
RUN wget --progress=bar:force:noscroll https://github.com/MLIR-China/stage/releases/download/nightly-tag-2022-10-08-1203/llvm-install-MinSizeRel.zip && \
    unzip llvm-install.zip -d /llvm-install && \
    rm llvm-install.zip
ENV LLVM_CONFIG_PATH="/llvm-install/bin/llvm-config"

# setup zig
RUN wget --progress=bar:force:noscroll https://ziglang.org/download/0.9.1/zig-linux-x86_64-0.9.1.tar.xz -O zig-install.tar.xz && \
    mkdir /zig-install && \
    tar xvf zig-install.tar.xz -C /zig-install --strip-components 1 && \
    rm zig-install.tar.xz
ENV PATH "/zig-install:${PATH}"

FROM base as test

WORKDIR /beaver

# Install hex and rebar
RUN mix local.hex --force && \
    mix local.rebar --force

# Build for production
ENV MIX_ENV=prod

# Install mix dependencies
COPY mix.exs mix.lock ./
COPY config config

# Make sure it compiles in docker
RUN mix do deps.get, deps.compile
RUN mix do compile

FROM base
