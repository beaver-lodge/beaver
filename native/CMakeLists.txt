cmake_minimum_required(VERSION 3.13.4)
set(CMAKE_BUILD_TYPE
    Release
    CACHE STRING "")
project(mlir-beaver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_INSTALL_MESSAGE
    LAZY
    CACHE STRING "" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "" FORCE)
include(pre-built-llvm.cmake)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(aggregate)
