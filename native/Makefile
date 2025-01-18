LLVM_CONFIG_PATH ?= llvm-config
LLVM_LIB_DIR := $(shell ${LLVM_CONFIG_PATH} --libdir)
LLVM_CMAKE_DIR = $(LLVM_LIB_DIR)/cmake/llvm
MLIR_CMAKE_DIR = $(LLVM_LIB_DIR)/cmake/mlir
CMAKE_BUILD_DIR = ${MIX_APP_PATH}/cmake_build
NATIVE_INSTALL_DIR = ${MIX_APP_PATH}/priv
MLIR_INCLUDE_DIR = ${LLVM_LIB_DIR}/../include
BUILD_STAMP = ${CMAKE_BUILD_DIR}/CMakeCache.txt

all: zig_build

zig_translate:
	mkdir -p ${NATIVE_INSTALL_DIR}
	zig translate-c include/mlir-c/Beaver/wrapper.h -I include -I ${MLIR_INCLUDE_DIR} | tee ${MIX_APP_PATH}/wrapper.h.zig | elixir gen_wrapper.exs --elixir ${NATIVE_INSTALL_DIR}/capi_functions.ex --zig src/wrapper.zig
	mv ${MIX_APP_PATH}/wrapper.h.zig src/wrapper.h.zig

${BUILD_STAMP}:
	cmake -G Ninja -B ${CMAKE_BUILD_DIR} -DLLVM_DIR=${LLVM_CMAKE_DIR} -DMLIR_DIR=${MLIR_CMAKE_DIR} -DCMAKE_INSTALL_PREFIX=${NATIVE_INSTALL_DIR}

cmake_build: ${BUILD_STAMP}
	cmake --build ${CMAKE_BUILD_DIR} --target install
	cmake -E copy_if_different ${LLVM_LIB_DIR}/libmlir_* ${NATIVE_INSTALL_DIR}/lib

zig_build: zig_translate cmake_build
	zig build --prefix ${NATIVE_INSTALL_DIR} --search-prefix ${NATIVE_INSTALL_DIR} --search-prefix ${LLVM_LIB_DIR}/.. --search-prefix ${ERTS_INCLUDE_DIR}/.. -freference-trace

clean:
	rm -rf ${CMAKE_BUILD_DIR}
	rm -rf ${NATIVE_INSTALL_DIR}
	rm -rf .zig-cache
