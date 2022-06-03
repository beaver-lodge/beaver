add_mlir_dialect_library(MLIRElixir
        ElixirDialect.cpp
        ElixirOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/csrc/include

        DEPENDS
        MLIRElixirOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
