set -e
export MIX_ENV=prod
mix compile.elixir_make
mix compile
LIB_FINAL_SO=$(ls _build/${MIX_ENV}/lib/beaver/native_install/lib | grep -E "libbeaver.+so")
LIB_FINAL_NAME=${LIB_FINAL_SO}.tar.gz
tar --dereference -cvzf ${LIB_FINAL_NAME} \
    -C $PWD/_build/${MIX_ENV}/lib/beaver/native_install/lib $(cd $PWD/_build/${MIX_ENV}/lib/beaver/native_install/lib && ls *.so) \
    -C $PWD/_build/${MIX_ENV}/lib/beaver/native_install/lib $(cd $PWD/_build/${MIX_ENV}/lib/beaver/native_install/lib && ls *.dylib) \
    -C $PWD/_build/${MIX_ENV}/lib/beaver/native_install $(cd $PWD/_build/${MIX_ENV}/lib/beaver/native_install && ls *.ex)
