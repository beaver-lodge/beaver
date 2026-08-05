MIX_ENV ?= dev
MIX_APP_PATH ?= _build/$(MIX_ENV)/lib/beaver
BUILD_PRIV := $(MIX_APP_PATH)/priv

all: zig_build

prepare_build_priv:
	mkdir -p $(MIX_APP_PATH)
	if [ -L "$(BUILD_PRIV)" ]; then rm "$(BUILD_PRIV)"; fi
	mkdir -p "$(BUILD_PRIV)"

zig_build: prepare_build_priv
	zig build -p $(BUILD_PRIV) --search-prefix ${ERTS_INCLUDE_DIR}/..

clean:
	rm -rf .zig-cache
