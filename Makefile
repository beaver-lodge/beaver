LLVM_CONFIG_PATH ?= $($(shell echo $$LLVM_CONFIG_PATH), llvm-config)
LLVM_LIB_DIR := $(shell ${LLVM_CONFIG_PATH} --libdir)
LLVM_CMAKE_DIR = $(LLVM_LIB_DIR)/cmake/llvm
MLIR_CMAKE_DIR = $(LLVM_LIB_DIR)/cmake/mlir
CMAKE_BUILD_DIR = ${MIX_APP_PATH}/cmake_build
NATIVE_INSTALL_DIR = ${MIX_APP_PATH}/priv
MLIR_INCLUDE_DIR = ${LLVM_LIB_DIR}/../include
BEAVER_INCLUDE_DIR = native/include
ZIG_CACHE_DIR = ${MIX_APP_PATH}/zig_cache
.PHONY: all zig_build cmake_build

all: zig_build

zig_translate:
	mkdir -p ${NATIVE_INSTALL_DIR}
	zig translate-c ${BEAVER_INCLUDE_DIR}/mlir-c/Beaver/wrapper.h --cache-dir ${ZIG_CACHE_DIR} \
		-I ${BEAVER_INCLUDE_DIR} \
		-I ${MLIR_INCLUDE_DIR} | tee native/src/wrapper.h.zig | elixir scripts/update_generated.exs \
			--elixir ${NATIVE_INSTALL_DIR}/capi_functions.ex \
			--zig native/src/wrapper.zig
zig_build:
	( $(MAKE) zig_translate & $(MAKE) cmake_build & wait )
	zig build --build-file native/build.zig --cache-dir ${ZIG_CACHE_DIR} \
	  --prefix ${NATIVE_INSTALL_DIR} \
		--search-prefix ${NATIVE_INSTALL_DIR} \
		--search-prefix ${LLVM_LIB_DIR}/.. \
		--search-prefix ${ERTS_INCLUDE_DIR}/.. \
		-freference-trace
cmake_config:
	cmake -G Ninja -S native -B ${CMAKE_BUILD_DIR} -DLLVM_DIR=${LLVM_CMAKE_DIR} -DMLIR_DIR=${MLIR_CMAKE_DIR} -DCMAKE_INSTALL_PREFIX=${NATIVE_INSTALL_DIR}
cmake_build: cmake_config
	cmake --build ${CMAKE_BUILD_DIR} --target install
	cp -v ${LLVM_LIB_DIR}/libmlir_* ${NATIVE_INSTALL_DIR}/lib

clean:
	rm -rf ${CMAKE_BUILD_DIR}
	rm -rf ${NATIVE_INSTALL_DIR}
