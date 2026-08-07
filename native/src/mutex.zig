//! A small portable mutex for native code that must be usable from arbitrary
//! worker threads (for example MLIR diagnostic callbacks). Windows has no
//! pthreads in the emulator context, so the SRWLOCK API is used there.
const std = @import("std");
const builtin = @import("builtin");
const windows = std.os.windows;

extern "kernel32" fn AcquireSRWLockExclusive(srwlock: *windows.SRWLOCK) void;
extern "kernel32" fn ReleaseSRWLockExclusive(srwlock: *windows.SRWLOCK) void;

pub const Mutex = if (builtin.os.tag == .windows)
    struct {
        srw: windows.SRWLOCK = .{},

        pub fn lock(self: *Mutex) void {
            AcquireSRWLockExclusive(&self.srw);
        }

        pub fn unlock(self: *Mutex) void {
            ReleaseSRWLockExclusive(&self.srw);
        }

        pub fn destroy(self: *Mutex) void {
            _ = self;
        }
    }
else
    struct {
        raw: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,

        pub fn lock(self: *Mutex) void {
            if (std.c.pthread_mutex_lock(&self.raw) != .SUCCESS)
                @panic("failed to lock mutex");
        }

        pub fn unlock(self: *Mutex) void {
            if (std.c.pthread_mutex_unlock(&self.raw) != .SUCCESS)
                @panic("failed to unlock mutex");
        }

        pub fn destroy(self: *Mutex) void {
            if (std.c.pthread_mutex_destroy(&self.raw) != .SUCCESS)
                @panic("failed to destroy mutex");
        }
    };
