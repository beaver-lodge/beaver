set(LLVM_LINK_COMPONENTS Demangle Support TableGen)

set(LIBS MLIRIR MLIRPDLLODS MLIRTableGen)

add_tablegen(
  ods-extract
  MLIR_PDLL
  EXPORT
  MLIR
  mlir-pdll.cpp
  DEPENDS
  ${LIBS})

target_link_libraries(ods-extract PRIVATE ${LIBS})

mlir_check_all_link_libraries(ods-extract)

install(TARGETS ods-extract RUNTIME DESTINATION tools)
