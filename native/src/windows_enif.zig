// Generated on 2026-08-06 from erl_nif.h (OTP 27.x) by translate-c with
// STATIC_ERLANG_NIF=1. On Windows the BEAM does not export enif_* functions:
// they are only reachable through the TWinDynNifCallbacks table passed to
// nif_init. These exports forward each enif_* import to that table so Zig
// code can call them through the usual erl_nif namespace.
const e = @import("kinda").erl_nif;

// The BEAM hands Windows NIFs the enif_* function table through nif_init;
// this global is what the forwards below read. It lives in this artifact so
// every native library that imports enif_* resolves the same table.
export var WinDynNifCallbacks: e.TWinDynNifCallbacks = .{};

export fn enif_free(ptr: ?*anyopaque) void {
    e.WinDynNifCallbacks.enif_free.?(ptr);
}

export fn enif_mutex_destroy(mtx: ?*e.ErlNifMutex) void {
    e.WinDynNifCallbacks.enif_mutex_destroy.?(mtx);
}

export fn enif_cond_destroy(cnd: ?*e.ErlNifCond) void {
    e.WinDynNifCallbacks.enif_cond_destroy.?(cnd);
}

export fn enif_rwlock_destroy(rwlck: ?*e.ErlNifRWLock) void {
    e.WinDynNifCallbacks.enif_rwlock_destroy.?(rwlck);
}

export fn enif_thread_opts_destroy(opts: [*c]e.ErlNifThreadOpts) void {
    e.WinDynNifCallbacks.enif_thread_opts_destroy.?(opts);
}

export fn enif_ioq_destroy(q: ?*e.ErlNifIOQueue) void {
    e.WinDynNifCallbacks.enif_ioq_destroy.?(q);
}

export fn enif_priv_data(arg0: ?*e.ErlNifEnv) ?*anyopaque {
    return e.WinDynNifCallbacks.enif_priv_data.?(arg0);
}

export fn enif_alloc(size: usize) ?*anyopaque {
    return e.WinDynNifCallbacks.enif_alloc.?(size);
}

export fn enif_is_atom(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_atom.?(arg0, term);
}

export fn enif_is_binary(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_binary.?(arg0, term);
}

export fn enif_is_ref(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_ref.?(arg0, term);
}

export fn enif_inspect_binary(arg0: ?*e.ErlNifEnv, bin_term: e.ERL_NIF_TERM, bin: [*c]e.ErlNifBinary) c_int {
    return e.WinDynNifCallbacks.enif_inspect_binary.?(arg0, bin_term, bin);
}

export fn enif_alloc_binary(size: usize, bin: [*c]e.ErlNifBinary) c_int {
    return e.WinDynNifCallbacks.enif_alloc_binary.?(size, bin);
}

export fn enif_realloc_binary(bin: [*c]e.ErlNifBinary, size: usize) c_int {
    return e.WinDynNifCallbacks.enif_realloc_binary.?(bin, size);
}

export fn enif_release_binary(bin: [*c]e.ErlNifBinary) void {
    e.WinDynNifCallbacks.enif_release_binary.?(bin);
}

export fn enif_get_int(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, ip: [*c]c_int) c_int {
    return e.WinDynNifCallbacks.enif_get_int.?(arg0, term, ip);
}

export fn enif_get_ulong(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, ip: [*c]c_ulong) c_int {
    return e.WinDynNifCallbacks.enif_get_ulong.?(arg0, term, ip);
}

export fn enif_get_double(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, dp: [*c]f64) c_int {
    return e.WinDynNifCallbacks.enif_get_double.?(arg0, term, dp);
}

export fn enif_get_list_cell(env: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, head: [*c]e.ERL_NIF_TERM, tail: [*c]e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_get_list_cell.?(env, term, head, tail);
}

export fn enif_get_tuple(env: ?*e.ErlNifEnv, tpl: e.ERL_NIF_TERM, arity: [*c]c_int, array: [*c][*c]const e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_get_tuple.?(env, tpl, arity, array);
}

export fn enif_is_identical(lhs: e.ERL_NIF_TERM, rhs: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_identical.?(lhs, rhs);
}

export fn enif_compare(lhs: e.ERL_NIF_TERM, rhs: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_compare.?(lhs, rhs);
}

export fn enif_make_binary(env: ?*e.ErlNifEnv, bin: [*c]e.ErlNifBinary) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_binary.?(env, bin);
}

export fn enif_make_badarg(env: ?*e.ErlNifEnv) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_badarg.?(env);
}

export fn enif_make_int(env: ?*e.ErlNifEnv, i: c_int) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_int.?(env, i);
}

export fn enif_make_ulong(env: ?*e.ErlNifEnv, i: c_ulong) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_ulong.?(env, i);
}

export fn enif_make_double(env: ?*e.ErlNifEnv, d: f64) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_double.?(env, d);
}

export fn enif_make_atom(env: ?*e.ErlNifEnv, name: [*c]const u8) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_atom.?(env, name);
}

export fn enif_make_existing_atom(env: ?*e.ErlNifEnv, name: [*c]const u8, atom: [*c]e.ERL_NIF_TERM, arg3: e.ErlNifCharEncoding) c_int {
    return e.WinDynNifCallbacks.enif_make_existing_atom.?(env, name, atom, arg3);
}

export fn enif_make_tuple(env: ?*e.ErlNifEnv, cnt: c_uint, t1: e.ERL_NIF_TERM, t2: e.ERL_NIF_TERM) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_tuple.?(env, cnt, t1, t2);
}

export fn enif_make_list(env: ?*e.ErlNifEnv, cnt: c_uint) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_list.?(env, cnt);
}

export fn enif_make_list_cell(env: ?*e.ErlNifEnv, car: e.ERL_NIF_TERM, cdr: e.ERL_NIF_TERM) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_list_cell.?(env, car, cdr);
}

export fn enif_make_string(env: ?*e.ErlNifEnv, string: [*c]const u8, arg2: e.ErlNifCharEncoding) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_string.?(env, string, arg2);
}

export fn enif_make_ref(env: ?*e.ErlNifEnv) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_ref.?(env);
}

export fn enif_mutex_create(name: [*c]u8) ?*e.ErlNifMutex {
    return e.WinDynNifCallbacks.enif_mutex_create.?(name);
}

export fn enif_mutex_trylock(mtx: ?*e.ErlNifMutex) c_int {
    return e.WinDynNifCallbacks.enif_mutex_trylock.?(mtx);
}

export fn enif_mutex_lock(mtx: ?*e.ErlNifMutex) void {
    e.WinDynNifCallbacks.enif_mutex_lock.?(mtx);
}

export fn enif_mutex_unlock(mtx: ?*e.ErlNifMutex) void {
    e.WinDynNifCallbacks.enif_mutex_unlock.?(mtx);
}

export fn enif_cond_create(name: [*c]u8) ?*e.ErlNifCond {
    return e.WinDynNifCallbacks.enif_cond_create.?(name);
}

export fn enif_cond_signal(cnd: ?*e.ErlNifCond) void {
    e.WinDynNifCallbacks.enif_cond_signal.?(cnd);
}

export fn enif_cond_broadcast(cnd: ?*e.ErlNifCond) void {
    e.WinDynNifCallbacks.enif_cond_broadcast.?(cnd);
}

export fn enif_cond_wait(cnd: ?*e.ErlNifCond, mtx: ?*e.ErlNifMutex) void {
    e.WinDynNifCallbacks.enif_cond_wait.?(cnd, mtx);
}

export fn enif_rwlock_create(name: [*c]u8) ?*e.ErlNifRWLock {
    return e.WinDynNifCallbacks.enif_rwlock_create.?(name);
}

export fn enif_rwlock_tryrlock(rwlck: ?*e.ErlNifRWLock) c_int {
    return e.WinDynNifCallbacks.enif_rwlock_tryrlock.?(rwlck);
}

export fn enif_rwlock_rlock(rwlck: ?*e.ErlNifRWLock) void {
    e.WinDynNifCallbacks.enif_rwlock_rlock.?(rwlck);
}

export fn enif_rwlock_runlock(rwlck: ?*e.ErlNifRWLock) void {
    e.WinDynNifCallbacks.enif_rwlock_runlock.?(rwlck);
}

export fn enif_rwlock_tryrwlock(rwlck: ?*e.ErlNifRWLock) c_int {
    return e.WinDynNifCallbacks.enif_rwlock_tryrwlock.?(rwlck);
}

export fn enif_rwlock_rwlock(rwlck: ?*e.ErlNifRWLock) void {
    e.WinDynNifCallbacks.enif_rwlock_rwlock.?(rwlck);
}

export fn enif_rwlock_rwunlock(rwlck: ?*e.ErlNifRWLock) void {
    e.WinDynNifCallbacks.enif_rwlock_rwunlock.?(rwlck);
}

export fn enif_tsd_key_create(name: [*c]u8, key: [*c]e.ErlNifTSDKey) c_int {
    return e.WinDynNifCallbacks.enif_tsd_key_create.?(name, key);
}

export fn enif_tsd_key_destroy(key: e.ErlNifTSDKey) void {
    e.WinDynNifCallbacks.enif_tsd_key_destroy.?(key);
}

export fn enif_tsd_set(key: e.ErlNifTSDKey, data: ?*anyopaque) void {
    e.WinDynNifCallbacks.enif_tsd_set.?(key, data);
}

export fn enif_tsd_get(key: e.ErlNifTSDKey) ?*anyopaque {
    return e.WinDynNifCallbacks.enif_tsd_get.?(key);
}

export fn enif_thread_opts_create(name: [*c]u8) [*c]e.ErlNifThreadOpts {
    return e.WinDynNifCallbacks.enif_thread_opts_create.?(name);
}

export fn enif_thread_create(name: [*c]u8, tid: [*c]e.ErlNifTid, func: ?*const fn (?*anyopaque) callconv(.c) ?*anyopaque, args: ?*anyopaque, opts: [*c]e.ErlNifThreadOpts) c_int {
    return e.WinDynNifCallbacks.enif_thread_create.?(name, tid, func, args, opts);
}

export fn enif_thread_self() e.ErlNifTid {
    return e.WinDynNifCallbacks.enif_thread_self.?();
}

export fn enif_equal_tids(tid1: e.ErlNifTid, tid2: e.ErlNifTid) c_int {
    return e.WinDynNifCallbacks.enif_equal_tids.?(tid1, tid2);
}

export fn enif_thread_exit(resp: ?*anyopaque) void {
    e.WinDynNifCallbacks.enif_thread_exit.?(resp);
}

export fn enif_thread_join(arg0: e.ErlNifTid, respp: [*c]?*anyopaque) c_int {
    return e.WinDynNifCallbacks.enif_thread_join.?(arg0, respp);
}

export fn enif_realloc(ptr: ?*anyopaque, size: usize) ?*anyopaque {
    return e.WinDynNifCallbacks.enif_realloc.?(ptr, size);
}

export fn enif_system_info(sip: [*c]e.ErlNifSysInfo, si_size: usize) void {
    e.WinDynNifCallbacks.enif_system_info.?(sip, si_size);
}

export fn enif_fprintf(filep: ?*e.FILE, format: [*c]const u8) c_int {
    return e.WinDynNifCallbacks.enif_fprintf.?(filep, format);
}

export fn enif_inspect_iolist_as_binary(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, bin: [*c]e.ErlNifBinary) c_int {
    return e.WinDynNifCallbacks.enif_inspect_iolist_as_binary.?(arg0, term, bin);
}

export fn enif_make_sub_binary(arg0: ?*e.ErlNifEnv, bin_term: e.ERL_NIF_TERM, pos: usize, size: usize) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_sub_binary.?(arg0, bin_term, pos, size);
}

export fn enif_get_string(arg0: ?*e.ErlNifEnv, list: e.ERL_NIF_TERM, buf: [*c]u8, len: c_uint, arg4: e.ErlNifCharEncoding) c_int {
    return e.WinDynNifCallbacks.enif_get_string.?(arg0, list, buf, len, arg4);
}

export fn enif_get_atom(arg0: ?*e.ErlNifEnv, atom: e.ERL_NIF_TERM, buf: [*c]u8, len: c_uint, arg4: e.ErlNifCharEncoding) c_int {
    return e.WinDynNifCallbacks.enif_get_atom.?(arg0, atom, buf, len, arg4);
}

export fn enif_is_fun(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_fun.?(arg0, term);
}

export fn enif_is_pid(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_pid.?(arg0, term);
}

export fn enif_is_port(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_port.?(arg0, term);
}

export fn enif_get_uint(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, ip: [*c]c_uint) c_int {
    return e.WinDynNifCallbacks.enif_get_uint.?(arg0, term, ip);
}

export fn enif_get_long(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, ip: [*c]c_long) c_int {
    return e.WinDynNifCallbacks.enif_get_long.?(arg0, term, ip);
}

export fn enif_make_uint(arg0: ?*e.ErlNifEnv, i: c_uint) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_uint.?(arg0, i);
}

export fn enif_make_long(arg0: ?*e.ErlNifEnv, i: c_long) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_long.?(arg0, i);
}

export fn enif_make_tuple_from_array(arg0: ?*e.ErlNifEnv, arr: [*c]const e.ERL_NIF_TERM, cnt: c_uint) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_tuple_from_array.?(arg0, arr, cnt);
}

export fn enif_make_list_from_array(arg0: ?*e.ErlNifEnv, arr: [*c]const e.ERL_NIF_TERM, cnt: c_uint) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_list_from_array.?(arg0, arr, cnt);
}

export fn enif_is_empty_list(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_empty_list.?(arg0, term);
}

export fn enif_open_resource_type(arg0: ?*e.ErlNifEnv, module_str: [*c]const u8, name_str: [*c]const u8, dtor: ?*const fn (?*e.ErlNifEnv, ?*anyopaque) callconv(.c) void, flags: e.ErlNifResourceFlags, tried: [*c]e.ErlNifResourceFlags) ?*e.ErlNifResourceType {
    return e.WinDynNifCallbacks.enif_open_resource_type.?(arg0, module_str, name_str, dtor, flags, tried);
}

export fn enif_alloc_resource(@"type": ?*e.ErlNifResourceType, size: usize) ?*anyopaque {
    return e.WinDynNifCallbacks.enif_alloc_resource.?(@"type", size);
}

export fn enif_release_resource(obj: ?*anyopaque) void {
    e.WinDynNifCallbacks.enif_release_resource.?(obj);
}

export fn enif_make_resource(arg0: ?*e.ErlNifEnv, obj: ?*anyopaque) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_resource.?(arg0, obj);
}

export fn enif_get_resource(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, @"type": ?*e.ErlNifResourceType, objp: [*c]?*anyopaque) c_int {
    return e.WinDynNifCallbacks.enif_get_resource.?(arg0, term, @"type", objp);
}

export fn enif_sizeof_resource(obj: ?*anyopaque) usize {
    return e.WinDynNifCallbacks.enif_sizeof_resource.?(obj);
}

export fn enif_make_new_binary(arg0: ?*e.ErlNifEnv, size: usize, termp: [*c]e.ERL_NIF_TERM) [*c]u8 {
    return e.WinDynNifCallbacks.enif_make_new_binary.?(arg0, size, termp);
}

export fn enif_is_list(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_list.?(arg0, term);
}

export fn enif_is_tuple(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_tuple.?(arg0, term);
}

export fn enif_get_atom_length(arg0: ?*e.ErlNifEnv, atom: e.ERL_NIF_TERM, len: [*c]c_uint, arg3: e.ErlNifCharEncoding) c_int {
    return e.WinDynNifCallbacks.enif_get_atom_length.?(arg0, atom, len, arg3);
}

export fn enif_get_list_length(env: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, len: [*c]c_uint) c_int {
    return e.WinDynNifCallbacks.enif_get_list_length.?(env, term, len);
}

export fn enif_make_atom_len(env: ?*e.ErlNifEnv, name: [*c]const u8, len: usize) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_atom_len.?(env, name, len);
}

export fn enif_make_existing_atom_len(env: ?*e.ErlNifEnv, name: [*c]const u8, len: usize, atom: [*c]e.ERL_NIF_TERM, arg4: e.ErlNifCharEncoding) c_int {
    return e.WinDynNifCallbacks.enif_make_existing_atom_len.?(env, name, len, atom, arg4);
}

export fn enif_make_string_len(env: ?*e.ErlNifEnv, string: [*c]const u8, len: usize, arg3: e.ErlNifCharEncoding) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_string_len.?(env, string, len, arg3);
}

export fn enif_alloc_env() ?*e.ErlNifEnv {
    return e.WinDynNifCallbacks.enif_alloc_env.?();
}

export fn enif_free_env(env: ?*e.ErlNifEnv) void {
    e.WinDynNifCallbacks.enif_free_env.?(env);
}

export fn enif_clear_env(env: ?*e.ErlNifEnv) void {
    e.WinDynNifCallbacks.enif_clear_env.?(env);
}

export fn enif_send(env: ?*e.ErlNifEnv, to_pid: [*c]const e.ErlNifPid, msg_env: ?*e.ErlNifEnv, msg: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_send.?(env, to_pid, msg_env, msg);
}

export fn enif_make_copy(dst_env: ?*e.ErlNifEnv, src_term: e.ERL_NIF_TERM) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_copy.?(dst_env, src_term);
}

export fn enif_self(caller_env: ?*e.ErlNifEnv, pid: [*c]e.ErlNifPid) [*c]e.ErlNifPid {
    return e.WinDynNifCallbacks.enif_self.?(caller_env, pid);
}

export fn enif_get_local_pid(env: ?*e.ErlNifEnv, arg1: e.ERL_NIF_TERM, pid: [*c]e.ErlNifPid) c_int {
    return e.WinDynNifCallbacks.enif_get_local_pid.?(env, arg1, pid);
}

export fn enif_keep_resource(obj: ?*anyopaque) void {
    e.WinDynNifCallbacks.enif_keep_resource.?(obj);
}

export fn enif_make_resource_binary(arg0: ?*e.ErlNifEnv, obj: ?*anyopaque, data: ?*const anyopaque, size: usize) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_resource_binary.?(arg0, obj, data, size);
}

export fn enif_is_exception(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_exception.?(arg0, term);
}

export fn enif_make_reverse_list(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, list: [*c]e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_make_reverse_list.?(arg0, term, list);
}

export fn enif_is_number(arg0: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_number.?(arg0, term);
}

export fn enif_dlopen(lib: [*c]const u8, err_handler: ?*const fn (?*anyopaque, [*c]const u8) callconv(.c) void, err_arg: ?*anyopaque) ?*anyopaque {
    return e.WinDynNifCallbacks.enif_dlopen.?(lib, err_handler, err_arg);
}

export fn enif_dlsym(handle: ?*anyopaque, symbol: [*c]const u8, err_handler: ?*const fn (?*anyopaque, [*c]const u8) callconv(.c) void, err_arg: ?*anyopaque) ?*anyopaque {
    return e.WinDynNifCallbacks.enif_dlsym.?(handle, symbol, err_handler, err_arg);
}

export fn enif_consume_timeslice(arg0: ?*e.ErlNifEnv, percent: c_int) c_int {
    return e.WinDynNifCallbacks.enif_consume_timeslice.?(arg0, percent);
}

export fn enif_is_map(env: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_is_map.?(env, term);
}

export fn enif_get_map_size(env: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, size: [*c]usize) c_int {
    return e.WinDynNifCallbacks.enif_get_map_size.?(env, term, size);
}

export fn enif_make_new_map(env: ?*e.ErlNifEnv) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_new_map.?(env);
}

export fn enif_make_map_put(env: ?*e.ErlNifEnv, map_in: e.ERL_NIF_TERM, key: e.ERL_NIF_TERM, value: e.ERL_NIF_TERM, map_out: [*c]e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_make_map_put.?(env, map_in, key, value, map_out);
}

export fn enif_get_map_value(env: ?*e.ErlNifEnv, map: e.ERL_NIF_TERM, key: e.ERL_NIF_TERM, value: [*c]e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_get_map_value.?(env, map, key, value);
}

export fn enif_make_map_update(env: ?*e.ErlNifEnv, map_in: e.ERL_NIF_TERM, key: e.ERL_NIF_TERM, value: e.ERL_NIF_TERM, map_out: [*c]e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_make_map_update.?(env, map_in, key, value, map_out);
}

export fn enif_make_map_remove(env: ?*e.ErlNifEnv, map_in: e.ERL_NIF_TERM, key: e.ERL_NIF_TERM, map_out: [*c]e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_make_map_remove.?(env, map_in, key, map_out);
}

export fn enif_map_iterator_create(env: ?*e.ErlNifEnv, map: e.ERL_NIF_TERM, iter: [*c]e.ErlNifMapIterator, entry: e.ErlNifMapIteratorEntry) c_int {
    return e.WinDynNifCallbacks.enif_map_iterator_create.?(env, map, iter, entry);
}

export fn enif_map_iterator_destroy(env: ?*e.ErlNifEnv, iter: [*c]e.ErlNifMapIterator) void {
    e.WinDynNifCallbacks.enif_map_iterator_destroy.?(env, iter);
}

export fn enif_map_iterator_is_head(env: ?*e.ErlNifEnv, iter: [*c]e.ErlNifMapIterator) c_int {
    return e.WinDynNifCallbacks.enif_map_iterator_is_head.?(env, iter);
}

export fn enif_map_iterator_is_tail(env: ?*e.ErlNifEnv, iter: [*c]e.ErlNifMapIterator) c_int {
    return e.WinDynNifCallbacks.enif_map_iterator_is_tail.?(env, iter);
}

export fn enif_map_iterator_next(env: ?*e.ErlNifEnv, iter: [*c]e.ErlNifMapIterator) c_int {
    return e.WinDynNifCallbacks.enif_map_iterator_next.?(env, iter);
}

export fn enif_map_iterator_prev(env: ?*e.ErlNifEnv, iter: [*c]e.ErlNifMapIterator) c_int {
    return e.WinDynNifCallbacks.enif_map_iterator_prev.?(env, iter);
}

export fn enif_map_iterator_get_pair(env: ?*e.ErlNifEnv, iter: [*c]e.ErlNifMapIterator, key: [*c]e.ERL_NIF_TERM, value: [*c]e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_map_iterator_get_pair.?(env, iter, key, value);
}

export fn enif_schedule_nif(arg0: ?*e.ErlNifEnv, arg1: [*c]const u8, arg2: c_int, arg3: ?*const fn (?*e.ErlNifEnv, c_int, [*c]const e.ERL_NIF_TERM) callconv(.c) e.ERL_NIF_TERM, arg4: c_int, arg5: [*c]const e.ERL_NIF_TERM) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_schedule_nif.?(arg0, arg1, arg2, arg3, arg4, arg5);
}

export fn enif_has_pending_exception(env: ?*e.ErlNifEnv, reason: [*c]e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_has_pending_exception.?(env, reason);
}

export fn enif_raise_exception(env: ?*e.ErlNifEnv, reason: e.ERL_NIF_TERM) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_raise_exception.?(env, reason);
}

export fn enif_getenv(key: [*c]const u8, value: [*c]u8, value_size: [*c]usize) c_int {
    return e.WinDynNifCallbacks.enif_getenv.?(key, value, value_size);
}

export fn enif_monotonic_time(arg0: e.ErlNifTimeUnit) e.ErlNifTime {
    return e.WinDynNifCallbacks.enif_monotonic_time.?(arg0);
}

export fn enif_time_offset(arg0: e.ErlNifTimeUnit) e.ErlNifTime {
    return e.WinDynNifCallbacks.enif_time_offset.?(arg0);
}

export fn enif_convert_time_unit(arg0: e.ErlNifTime, arg1: e.ErlNifTimeUnit, arg2: e.ErlNifTimeUnit) e.ErlNifTime {
    return e.WinDynNifCallbacks.enif_convert_time_unit.?(arg0, arg1, arg2);
}

export fn enif_now_time(env: ?*e.ErlNifEnv) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_now_time.?(env);
}

export fn enif_cpu_time(env: ?*e.ErlNifEnv) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_cpu_time.?(env);
}

export fn enif_make_unique_integer(env: ?*e.ErlNifEnv, properties: e.ErlNifUniqueInteger) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_unique_integer.?(env, properties);
}

export fn enif_is_current_process_alive(env: ?*e.ErlNifEnv) c_int {
    return e.WinDynNifCallbacks.enif_is_current_process_alive.?(env);
}

export fn enif_is_process_alive(env: ?*e.ErlNifEnv, pid: [*c]e.ErlNifPid) c_int {
    return e.WinDynNifCallbacks.enif_is_process_alive.?(env, pid);
}

export fn enif_is_port_alive(env: ?*e.ErlNifEnv, port_id: [*c]e.ErlNifPort) c_int {
    return e.WinDynNifCallbacks.enif_is_port_alive.?(env, port_id);
}

export fn enif_get_local_port(env: ?*e.ErlNifEnv, arg1: e.ERL_NIF_TERM, port_id: [*c]e.ErlNifPort) c_int {
    return e.WinDynNifCallbacks.enif_get_local_port.?(env, arg1, port_id);
}

export fn enif_term_to_binary(env: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, bin: [*c]e.ErlNifBinary) c_int {
    return e.WinDynNifCallbacks.enif_term_to_binary.?(env, term, bin);
}

export fn enif_binary_to_term(env: ?*e.ErlNifEnv, data: [*c]const u8, sz: usize, term: [*c]e.ERL_NIF_TERM, opts: c_uint) usize {
    return e.WinDynNifCallbacks.enif_binary_to_term.?(env, data, sz, term, opts);
}

export fn enif_port_command(env: ?*e.ErlNifEnv, to_port: [*c]const e.ErlNifPort, msg_env: ?*e.ErlNifEnv, msg: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_port_command.?(env, to_port, msg_env, msg);
}

export fn enif_thread_type() c_int {
    return e.WinDynNifCallbacks.enif_thread_type.?();
}

export fn enif_snprintf(buffer: [*c]u8, size: usize, format: [*c]const u8) c_int {
    return e.WinDynNifCallbacks.enif_snprintf.?(buffer, size, format);
}

export fn enif_select(env: ?*e.ErlNifEnv, ev: e.ErlNifEvent, flags: e.enum_ErlNifSelectFlags, obj: ?*anyopaque, pid: [*c]const e.ErlNifPid, ref: e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_select.?(env, ev, flags, obj, pid, ref);
}

export fn enif_open_resource_type_x(arg0: ?*e.ErlNifEnv, name_str: [*c]const u8, arg2: [*c]const e.ErlNifResourceTypeInit, flags: e.ErlNifResourceFlags, tried: [*c]e.ErlNifResourceFlags) ?*e.ErlNifResourceType {
    return e.WinDynNifCallbacks.enif_open_resource_type_x.?(arg0, name_str, arg2, flags, tried);
}

export fn enif_monitor_process(arg0: ?*e.ErlNifEnv, obj: ?*anyopaque, arg2: [*c]const e.ErlNifPid, monitor: [*c]e.ErlNifMonitor) c_int {
    return e.WinDynNifCallbacks.enif_monitor_process.?(arg0, obj, arg2, monitor);
}

export fn enif_demonitor_process(arg0: ?*e.ErlNifEnv, obj: ?*anyopaque, monitor: [*c]const e.ErlNifMonitor) c_int {
    return e.WinDynNifCallbacks.enif_demonitor_process.?(arg0, obj, monitor);
}

export fn enif_compare_monitors(arg0: [*c]const e.ErlNifMonitor, arg1: [*c]const e.ErlNifMonitor) c_int {
    return e.WinDynNifCallbacks.enif_compare_monitors.?(arg0, arg1);
}

export fn enif_hash(@"type": e.ErlNifHash, term: e.ERL_NIF_TERM, salt: e.ErlNifUInt64) e.ErlNifUInt64 {
    return e.WinDynNifCallbacks.enif_hash.?(@"type", term, salt);
}

export fn enif_whereis_pid(env: ?*e.ErlNifEnv, name: e.ERL_NIF_TERM, pid: [*c]e.ErlNifPid) c_int {
    return e.WinDynNifCallbacks.enif_whereis_pid.?(env, name, pid);
}

export fn enif_whereis_port(env: ?*e.ErlNifEnv, name: e.ERL_NIF_TERM, port: [*c]e.ErlNifPort) c_int {
    return e.WinDynNifCallbacks.enif_whereis_port.?(env, name, port);
}

export fn enif_ioq_create(opts: e.ErlNifIOQueueOpts) ?*e.ErlNifIOQueue {
    return e.WinDynNifCallbacks.enif_ioq_create.?(opts);
}

export fn enif_ioq_enq_binary(q: ?*e.ErlNifIOQueue, bin: [*c]e.ErlNifBinary, skip: usize) c_int {
    return e.WinDynNifCallbacks.enif_ioq_enq_binary.?(q, bin, skip);
}

export fn enif_ioq_enqv(q: ?*e.ErlNifIOQueue, iov: [*c]e.ErlNifIOVec, skip: usize) c_int {
    return e.WinDynNifCallbacks.enif_ioq_enqv.?(q, iov, skip);
}

export fn enif_ioq_size(q: ?*e.ErlNifIOQueue) usize {
    return e.WinDynNifCallbacks.enif_ioq_size.?(q);
}

export fn enif_ioq_deq(q: ?*e.ErlNifIOQueue, count: usize, size: [*c]usize) c_int {
    return e.WinDynNifCallbacks.enif_ioq_deq.?(q, count, size);
}

export fn enif_ioq_peek(q: ?*e.ErlNifIOQueue, iovlen: [*c]c_int) [*c]e.SysIOVec {
    return e.WinDynNifCallbacks.enif_ioq_peek.?(q, iovlen);
}

export fn enif_inspect_iovec(env: ?*e.ErlNifEnv, max_length: usize, iovec_term: e.ERL_NIF_TERM, tail: [*c]e.ERL_NIF_TERM, iovec: [*c][*c]e.ErlNifIOVec) c_int {
    return e.WinDynNifCallbacks.enif_inspect_iovec.?(env, max_length, iovec_term, tail, iovec);
}

export fn enif_free_iovec(iov: [*c]e.ErlNifIOVec) void {
    e.WinDynNifCallbacks.enif_free_iovec.?(iov);
}

export fn enif_ioq_peek_head(env: ?*e.ErlNifEnv, q: ?*e.ErlNifIOQueue, size: [*c]usize, head: [*c]e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_ioq_peek_head.?(env, q, size, head);
}

export fn enif_mutex_name(arg0: ?*e.ErlNifMutex) [*c]u8 {
    return e.WinDynNifCallbacks.enif_mutex_name.?(arg0);
}

export fn enif_cond_name(arg0: ?*e.ErlNifCond) [*c]u8 {
    return e.WinDynNifCallbacks.enif_cond_name.?(arg0);
}

export fn enif_rwlock_name(arg0: ?*e.ErlNifRWLock) [*c]u8 {
    return e.WinDynNifCallbacks.enif_rwlock_name.?(arg0);
}

export fn enif_thread_name(arg0: e.ErlNifTid) [*c]u8 {
    return e.WinDynNifCallbacks.enif_thread_name.?(arg0);
}

export fn enif_vfprintf(arg0: ?*e.FILE, fmt: [*c]const u8, arg2: e.va_list) c_int {
    return e.WinDynNifCallbacks.enif_vfprintf.?(arg0, fmt, arg2);
}

export fn enif_vsnprintf(arg0: [*c]u8, arg1: usize, fmt: [*c]const u8, arg3: e.va_list) c_int {
    return e.WinDynNifCallbacks.enif_vsnprintf.?(arg0, arg1, fmt, arg3);
}

export fn enif_make_map_from_arrays(env: ?*e.ErlNifEnv, keys: [*c]e.ERL_NIF_TERM, values: [*c]e.ERL_NIF_TERM, cnt: usize, map_out: [*c]e.ERL_NIF_TERM) c_int {
    return e.WinDynNifCallbacks.enif_make_map_from_arrays.?(env, keys, values, cnt, map_out);
}

export fn enif_select_x(env: ?*e.ErlNifEnv, ev: e.ErlNifEvent, flags: e.enum_ErlNifSelectFlags, obj: ?*anyopaque, pid: [*c]const e.ErlNifPid, msg: e.ERL_NIF_TERM, msg_env: ?*e.ErlNifEnv) c_int {
    return e.WinDynNifCallbacks.enif_select_x.?(env, ev, flags, obj, pid, msg, msg_env);
}

export fn enif_make_monitor_term(env: ?*e.ErlNifEnv, arg1: [*c]const e.ErlNifMonitor) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_monitor_term.?(env, arg1);
}

export fn enif_set_pid_undefined(pid: [*c]e.ErlNifPid) void {
    e.WinDynNifCallbacks.enif_set_pid_undefined.?(pid);
}

export fn enif_is_pid_undefined(pid: [*c]const e.ErlNifPid) c_int {
    return e.WinDynNifCallbacks.enif_is_pid_undefined.?(pid);
}

export fn enif_term_type(env: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM) e.ErlNifTermType {
    return e.WinDynNifCallbacks.enif_term_type.?(env, term);
}

export fn enif_init_resource_type(arg0: ?*e.ErlNifEnv, name_str: [*c]const u8, arg2: [*c]const e.ErlNifResourceTypeInit, flags: e.ErlNifResourceFlags, tried: [*c]e.ErlNifResourceFlags) ?*e.ErlNifResourceType {
    return e.WinDynNifCallbacks.enif_init_resource_type.?(arg0, name_str, arg2, flags, tried);
}

export fn enif_dynamic_resource_call(arg0: ?*e.ErlNifEnv, mod: e.ERL_NIF_TERM, name: e.ERL_NIF_TERM, rsrc: e.ERL_NIF_TERM, call_data: ?*anyopaque) c_int {
    return e.WinDynNifCallbacks.enif_dynamic_resource_call.?(arg0, mod, name, rsrc, call_data);
}

export fn enif_get_string_length(env: ?*e.ErlNifEnv, list: e.ERL_NIF_TERM, len: [*c]c_uint, encoding: e.ErlNifCharEncoding) c_int {
    return e.WinDynNifCallbacks.enif_get_string_length.?(env, list, len, encoding);
}

export fn enif_make_new_atom(env: ?*e.ErlNifEnv, name: [*c]const u8, atom: [*c]e.ERL_NIF_TERM, encoding: e.ErlNifCharEncoding) c_int {
    return e.WinDynNifCallbacks.enif_make_new_atom.?(env, name, atom, encoding);
}

export fn enif_make_new_atom_len(env: ?*e.ErlNifEnv, name: [*c]const u8, len: usize, atom: [*c]e.ERL_NIF_TERM, encoding: e.ErlNifCharEncoding) c_int {
    return e.WinDynNifCallbacks.enif_make_new_atom_len.?(env, name, len, atom, encoding);
}

// The MSVC emulator keeps 64-bit signedness distinct from c_long, so these
// are real callbacks there. On LP64 ABIs they alias the long variants and
// translate-c emits aliases instead, so the callbacks table has no fields.
export fn enif_get_int64(env: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, ip: [*c]e.ErlNifSInt64) c_int {
    return if (@hasField(e.TWinDynNifCallbacks, "enif_get_int64"))
        e.WinDynNifCallbacks.enif_get_int64.?(env, term, ip)
    else
        e.enif_get_long(env, term, @ptrCast(ip));
}

export fn enif_get_uint64(env: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, ip: [*c]e.ErlNifUInt64) c_int {
    return if (@hasField(e.TWinDynNifCallbacks, "enif_get_uint64"))
        e.WinDynNifCallbacks.enif_get_uint64.?(env, term, ip)
    else
        e.enif_get_ulong(env, term, @ptrCast(ip));
}

export fn enif_make_int64(env: ?*e.ErlNifEnv, value: e.ErlNifSInt64) e.ERL_NIF_TERM {
    return if (@hasField(e.TWinDynNifCallbacks, "enif_make_int64"))
        e.WinDynNifCallbacks.enif_make_int64.?(env, value)
    else
        e.enif_make_long(env, @intCast(value));
}

export fn enif_make_uint64(env: ?*e.ErlNifEnv, value: e.ErlNifUInt64) e.ERL_NIF_TERM {
    return if (@hasField(e.TWinDynNifCallbacks, "enif_make_uint64"))
        e.WinDynNifCallbacks.enif_make_uint64.?(env, value)
    else
        e.enif_make_ulong(env, @intCast(value));
}

export fn enif_set_option(env: ?*e.ErlNifEnv, opt: e.ErlNifOption) c_int {
    return e.WinDynNifCallbacks.enif_set_option.?(env, opt);
}
