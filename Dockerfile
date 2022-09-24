FROM livebook/livebook:latest

WORKDIR /beaver

# Install hex and rebar
RUN mix local.hex --force && \
    mix local.rebar --force

# Build for production
ENV MIX_ENV=prod

RUN wget --progress=bar:force:noscroll https://github.com/MLIR-China/stage/releases/download/nightly-tag-2022-08-23-0233/llvm-install.zip && \
    unzip llvm-install.zip -d /llvm-install && \
    rm llvm-install.zip
ENV LLVM_CONFIG_PATH="/llvm-install/bin/llvm-config"
RUN wget --progress=bar:force:noscroll https://ziglang.org/download/0.9.1/zig-linux-x86_64-0.9.1.tar.xz -O zig-install.tar.xz && \
    tar xvf zig-install.tar.xz -d /zig-install -C /zig && \
    rm zig-install.tar.xz
ENV PATH "/zig:${PATH}"

# Install mix dependencies
COPY mix.exs mix.lock ./
COPY config config
RUN mix do deps.get, deps.compile

COPY lib lib
RUN mix do compile, release beaver
