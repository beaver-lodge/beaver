LLVM_CONFIG_PATH ?= $($(shell echo $$LLVM_CONFIG_PATH), llvm-config)
LLVM_LIB_DIR := $(shell ${LLVM_CONFIG_PATH} --libdir)
LLVM_CMAKE_DIR = $(LLVM_LIB_DIR)/cmake/llvm
MLIR_CMAKE_DIR = $(LLVM_LIB_DIR)/cmake/mlir
CMAKE_BUILD_DIR = ${MIX_APP_PATH}/cmake_build
NATIVE_INSTALL_DIR = ${MIX_APP_PATH}/native_install
cmake_build:
	cmake -G Ninja -S native/mlir-c -B ${CMAKE_BUILD_DIR} -DLLVM_DIR=${LLVM_CMAKE_DIR} -DMLIR_DIR=${MLIR_CMAKE_DIR} -DCMAKE_INSTALL_PREFIX=${NATIVE_INSTALL_DIR}
	cmake --build ${CMAKE_BUILD_DIR} --target install

clean:
	rm -rf ${CMAKE_BUILD_DIR}
	rm -rf ${NATIVE_INSTALL_DIR}
