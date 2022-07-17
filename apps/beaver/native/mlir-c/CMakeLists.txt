cmake_minimum_required(VERSION 3.13.4)
set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "")
project(met)
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_INSTALL_MESSAGE LAZY CACHE STRING "" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)
set(BUILD_LLVM FALSE CACHE BOOL "")
if(BUILD_LLVM)
    add_subdirectory(llvm-project)
else()
    include(pre-built-llvm.cmake)
endif(BUILD_LLVM)
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(nif)

add_custom_target(symlink_compile_commands_json ALL)
add_custom_command(TARGET symlink_compile_commands_json
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" ${PROJECT_SOURCE_DIR}/compile_commands.json
    DEPENDS compile_commands.json
    VERBATIM ON
)
