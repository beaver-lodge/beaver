#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
#[rustler::nif]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[rustler::nif]
fn registered_ops() -> Vec<(String, String)> {
    let mut ret = [].to_vec();
    unsafe {
        let ctx = mlirContextCreate();
        mlirRegisterAllDialects(ctx);
        let num_op = beaverGetNumRegisteredOperations(ctx);
        for i in 0..num_op {
            let registered_op_name = beaverGetRegisteredOperationName(ctx, i);
            let dialect_name = beaverRegisteredOperationNameGetDialectName(registered_op_name);
            let op_name = beaverRegisteredOperationNameGetOpName(registered_op_name);
            let dialect = std::str::from_utf8(std::slice::from_raw_parts(
                dialect_name.data as *const u8,
                dialect_name.length as usize,
            ));
            let op = std::str::from_utf8(std::slice::from_raw_parts(
                op_name.data as *const u8,
                op_name.length as usize,
            ));
            ret.push((dialect.unwrap().to_string(), op.unwrap().to_string()));
        }
        mlirContextDestroy(ctx);
    }
    return ret;
}

rustler::init!("Elixir.Beaver.MLIR.NIF", [add, registered_ops]);
