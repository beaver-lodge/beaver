FROM ghcr.io/livebook-dev/livebook:0.14.5
RUN apt-get upgrade -y \
  && apt-get update \
  && apt-get install --no-install-recommends -y \
    ninja-build \
    python3-pip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
# LLVM
COPY ./dev-requirements.txt /src/dev-requirements.txt
RUN python3 -m pip install -r /src/dev-requirements.txt && python3 -m pip cache purge
RUN ln -s $(python3 -c 'import mlir;print(mlir.__path__[0])') /usr/local/mlir
ENV PATH=/usr/local/mlir/bin:${PATH}
RUN llvm-config --version
# Zig
ARG ZIG_URL="https://ziglang.org/download/0.14.1/zig-aarch64-linux-0.14.1.tar.xz"
RUN wget "${ZIG_URL}" -O "zig-linux.tar.xz" \
  && tar Jxvf "zig-linux.tar.xz" -C /usr/local \
  && mv /usr/local/zig-linux-*-* /usr/local/zig-linux \
  && rm "zig-linux.tar.xz"
ENV PATH=/usr/local/zig-linux:${PATH}
RUN zig version
ENV ERL_FLAGS="+JMsingle true"
