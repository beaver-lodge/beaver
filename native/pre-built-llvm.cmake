set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
find_package(MLIR REQUIRED CONFIG)
if(LLVM_PACKAGE_VERSION VERSION_LESS "24.0.0")
  message(FATAL_ERROR "LLVM 24 main with the Index DataLayout fix is required")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/index-layout.mlir" [=[
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>} {
  func.func @sum(%x: index) -> index {
    %a = index.add %x, %x
    return %a : index
  }
}
]=])
execute_process(COMMAND "${LLVM_TOOLS_BINARY_DIR}/mlir-opt${CMAKE_EXECUTABLE_SUFFIX}"
  "${CMAKE_CURRENT_BINARY_DIR}/index-layout.mlir" "--convert-index-to-llvm"
  RESULT_VARIABLE _index_result OUTPUT_VARIABLE _index_output ERROR_VARIABLE _index_error TIMEOUT 30)
if(NOT "${_index_result}" STREQUAL "0" OR NOT _index_output MATCHES "index to i32")
  message(FATAL_ERROR "LLVM lacks the required Index DataLayout semantics (d1ec07e04314): ${_index_error}")
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(LLVM_VERSION_SUFFIX "")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
