LLVM_CONFIG_PATH ?= llvm-config
LLVM_LIB_DIR := $(shell ${LLVM_CONFIG_PATH} --libdir)
LLVM_CMAKE_DIR = $(LLVM_LIB_DIR)/cmake/llvm
MLIR_CMAKE_DIR = $(LLVM_LIB_DIR)/cmake/mlir
CMAKE_BUILD_DIR = ${MIX_APP_PATH}/cmake_build
NATIVE_INSTALL_DIR = ${MIX_APP_PATH}/priv

all: zig_build

zig_build:
	zig build --prefix ${NATIVE_INSTALL_DIR} \
		--search-prefix ${ERTS_INCLUDE_DIR}/.. \
		-freference-trace --verbose

clean:
	rm -rf ${CMAKE_BUILD_DIR}
	rm -rf ${NATIVE_INSTALL_DIR}
	rm -rf .zig-cache
