pub const Struct = extern struct {
    // const field_maker: type = fn(environment: env, []u8) !term;
    storage: std.SegmentedList(u8, 32),
    // makers: std.ArrayList(field_maker)
};
