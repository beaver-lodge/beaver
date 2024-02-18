LLVM_CONFIG_PATH ?= $($(shell echo $$LLVM_CONFIG_PATH), llvm-config)
LLVM_LIB_DIR := $(shell ${LLVM_CONFIG_PATH} --libdir)
LLVM_CMAKE_DIR = $(LLVM_LIB_DIR)/cmake/llvm
MLIR_CMAKE_DIR = $(LLVM_LIB_DIR)/cmake/mlir
CMAKE_BUILD_DIR = ${MIX_APP_PATH}/cmake_build
NATIVE_INSTALL_DIR = ${MIX_APP_PATH}/native_install
MLIR_INCLUDE_DIR = ${LLVM_LIB_DIR}/../include
BEAVER_INCLUDE_DIR = native/mlir-c/include
ZIG_CACHE_DIR = ${MIX_APP_PATH}/zig_cache
.PHONY: all zig_build cmake_build

all: zig_build

zig_build: cmake_build
	zig translate-c ${BEAVER_INCLUDE_DIR}/mlir-c/Beaver/wrapper.h --cache-dir ${ZIG_CACHE_DIR} \
		-I ${BEAVER_INCLUDE_DIR} \
		-I ${MLIR_INCLUDE_DIR} | elixir scripts/update_generated.exs
	cd native/mlir-zig-proj && zig build --cache-dir ${ZIG_CACHE_DIR} \
	  --prefix ${NATIVE_INSTALL_DIR} \
		--search-prefix ../mlir-c \
		--search-prefix ${NATIVE_INSTALL_DIR} \
		--search-prefix ${LLVM_LIB_DIR}/.. \
		--search-prefix ${ERTS_INCLUDE_DIR}/.. \
		-freference-trace \
		-DKINDA_LIB_NAME=beaver-v0.3.2-${KINDA_NIF_TARGET}
cmake_build:
	cmake -G Ninja -S native/mlir-c -B ${CMAKE_BUILD_DIR} -DLLVM_DIR=${LLVM_CMAKE_DIR} -DMLIR_DIR=${MLIR_CMAKE_DIR} -DCMAKE_INSTALL_PREFIX=${NATIVE_INSTALL_DIR}
	cmake --build ${CMAKE_BUILD_DIR} --target install

clean:
	rm -rf ${CMAKE_BUILD_DIR}
	rm -rf ${NATIVE_INSTALL_DIR}
