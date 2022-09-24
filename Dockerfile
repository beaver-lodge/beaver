FROM hexpm/elixir:1.14.0-erlang-24.3.4.2-debian-bullseye-20210902-slim AS build

RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y \
    build-essential git unzip wget ninja-build ca-certificates cmake zlib1g-dev libzstd-dev ccache && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /beaver

# Install hex and rebar
RUN mix local.hex --force && \
    mix local.rebar --force

# Build for production
ENV MIX_ENV=prod

RUN wget --progress=bar:force:noscroll https://github.com/MLIR-China/stage/releases/download/nightly-tag-2022-09-24-0520/llvm-install.zip && \
    unzip llvm-install.zip -d /llvm-install && \
    rm llvm-install.zip
ENV LLVM_CONFIG_PATH="/llvm-install/bin/llvm-config"
RUN wget --progress=bar:force:noscroll https://ziglang.org/download/0.9.1/zig-linux-x86_64-0.9.1.tar.xz -O zig-install.tar.xz && \
    mkdir /zig-install && \
    tar xvf zig-install.tar.xz -C /zig-install --strip-components 1 && \
    rm zig-install.tar.xz
ENV PATH "/zig-install:${PATH}"

# Install mix dependencies
COPY mix.exs mix.lock ./
COPY config config
COPY apps apps

RUN mix do deps.get, deps.compile
RUN mix do compile, release beaver
