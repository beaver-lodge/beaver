all: zig_build

zig_build:
	zig build -p ${MIX_APP_PATH}/priv --search-prefix ${ERTS_INCLUDE_DIR}/..

clean:
	rm -rf .zig-cache
