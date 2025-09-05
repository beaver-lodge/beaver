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
	mkdir -p ${MIX_APP_PATH}/include/mlir-c/Beaver
	elixir tools/wrapper/gen_header.exs --mlir-include-dir ${MLIR_INCLUDE_DIR} --output ${MIX_APP_PATH}/include/mlir-c/Beaver/wrapper.h
	zig cc -E -Xclang -ast-dump=json ${MIX_APP_PATH}/include/mlir-c/Beaver/wrapper.h -I include -I ${MLIR_INCLUDE_DIR} | \
		tee ${MIX_APP_PATH}/wrapper.h.clang.ast.json | \
		elixir tools/wrapper/gen.exs --elixir ${NATIVE_INSTALL_DIR}/capi_functions.ex --zig src/wrapper.zig

${BUILD_STAMP}:
	cmake -G Ninja -B ${CMAKE_BUILD_DIR} -DLLVM_DIR=${LLVM_CMAKE_DIR} -DMLIR_DIR=${MLIR_CMAKE_DIR} -DCMAKE_INSTALL_PREFIX=${NATIVE_INSTALL_DIR}

cmake_build: ${BUILD_STAMP}
	cmake --build ${CMAKE_BUILD_DIR} --target install
	cmake -E copy_if_different ${LLVM_LIB_DIR}/libmlir_* ${NATIVE_INSTALL_DIR}/lib

zig_build: zig_translate cmake_build
	elixir tools/ods-extract/dump_ods.exs --mlir-include-dir ${MLIR_INCLUDE_DIR} | xargs \
	${NATIVE_INSTALL_DIR}/tools/ods-extract -I ${MLIR_INCLUDE_DIR} \
		-I ${MLIR_INCLUDE_DIR}/mlir/Dialect/ArmSME/IR \
		-I ${MLIR_INCLUDE_DIR}/mlir/Dialect/IRDL/IR \
		-I ${MLIR_INCLUDE_DIR}/mlir/Dialect/UB/IR \
		-write-if-changed \
	-o ${MIX_APP_PATH}/ods_dump.json
	elixir tools/json_to_inspection.exs --input ${MIX_APP_PATH}/ods_dump.json --output ${NATIVE_INSTALL_DIR}/ods_dump.ex
	zig build --prefix ${NATIVE_INSTALL_DIR} \
		--search-prefix ${NATIVE_INSTALL_DIR} \
		--search-prefix ${LLVM_LIB_DIR}/.. \
		--search-prefix ${ERTS_INCLUDE_DIR}/.. \
		--search-prefix ${MIX_APP_PATH} \
		-freference-trace

clean:
	rm -rf ${CMAKE_BUILD_DIR}
	rm -rf ${NATIVE_INSTALL_DIR}
	rm -rf .zig-cache
