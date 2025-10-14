all: zig_build

zig_build:
	zig build --prefix ${MIX_APP_PATH}/priv \
		--search-prefix ${ERTS_INCLUDE_DIR}/.. \
		-freference-trace --verbose

clean:
	rm -rf .zig-cache
