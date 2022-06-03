use core::slice;
use libffi::middle::*;
use libffi::raw::ffi_type;
use libloading::{Library, Symbol};
use rustler::Atom;
use rustler::Encoder;
use rustler::Env;
use rustler::Error;
use rustler::ListIterator;
use rustler::LocalPid;
use rustler::NifResult;
use rustler::OwnedBinary;
use rustler::OwnedEnv;
use rustler::ResourceArc;
use rustler::Term;
use std::convert::TryInto;
use std::io::Read;
use std::io::Write;
use std::os::raw::c_void;
use std::sync::Condvar;
use std::sync::Mutex;
use std::thread;
#[derive(Debug, Clone)]
struct StructBuffer {
    pub data: Vec<u8>,
}

impl StructBuffer {
    fn new(size: usize) -> Self {
        let mut data = Vec::with_capacity(size);
        data.fill(0);
        StructBuffer { data: data }
    }
}
#[derive(Clone, Debug)]
enum TypeWrapper {
    Void,
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Size,
    F32,
    F64,
    Ptr,
    Struct(Vec<TypeWrapper>),
}

// TODO: rustler provide auto decoder for primitive types, so we have to add a wrapper to make it dynamic.
// There could be a performance penalty.
#[derive(Debug)]
enum ValueWrapper {
    Void,
    Bool(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Ptr(usize),
    Size(usize),
    Struct(StructBuffer),
    Closure(ClosureWrapper),
}

// this is the same as the impl from, with the exception that pointer is public so that we can assign a vector data ptr.
// TODO: use a more elegant way to do this. maybe a type to represent the data of a vector?
#[derive(Clone, Debug)]
#[repr(C)]
pub struct ArgWrapper(pub *mut c_void);

impl ArgWrapper {
    pub fn new<T>(r: &T) -> Self {
        ArgWrapper(r as *const T as *mut c_void)
    }
}

impl ValueWrapper {
    pub fn get_arg(&self) -> ArgWrapper {
        match self {
            ValueWrapper::Void => panic!("void type can't be used as arg"),
            ValueWrapper::Bool(v) => ArgWrapper::new(v),
            ValueWrapper::U8(v) => ArgWrapper::new(v),
            ValueWrapper::U16(v) => ArgWrapper::new(v),
            ValueWrapper::U32(v) => ArgWrapper::new(v),
            ValueWrapper::U64(v) => ArgWrapper::new(v),
            ValueWrapper::I8(v) => ArgWrapper::new(v),
            ValueWrapper::I16(v) => ArgWrapper::new(v),
            ValueWrapper::I32(v) => ArgWrapper::new(v),
            ValueWrapper::I64(v) => ArgWrapper::new(v),
            ValueWrapper::F32(v) => ArgWrapper::new(v),
            ValueWrapper::F64(v) => ArgWrapper::new(v),
            ValueWrapper::Ptr(v) => ArgWrapper::new(v),
            ValueWrapper::Size(v) => ArgWrapper::new(v),
            ValueWrapper::Struct(v) => {
                let p = v.data.as_ptr() as *mut c_void;
                ArgWrapper(p)
            }
            ValueWrapper::Closure(_) => {
                panic!("closure can only be used as argument directly, use it as pointer")
            }
        }
    }

    pub fn get_ptr(&self) -> *mut c_void {
        fn get_ptr_impl<T>(v: &T) -> *mut c_void {
            v as *const T as *mut c_void
        }
        match self {
            ValueWrapper::Void => panic!("can't get ptr of void"),
            ValueWrapper::Bool(v) => get_ptr_impl(v),
            ValueWrapper::U8(v) => get_ptr_impl(v),
            ValueWrapper::U16(v) => get_ptr_impl(v),
            ValueWrapper::U32(v) => get_ptr_impl(v),
            ValueWrapper::U64(v) => get_ptr_impl(v),
            ValueWrapper::I8(v) => get_ptr_impl(v),
            ValueWrapper::I16(v) => get_ptr_impl(v),
            ValueWrapper::I32(v) => get_ptr_impl(v),
            ValueWrapper::I64(v) => get_ptr_impl(v),
            ValueWrapper::F32(v) => get_ptr_impl(v),
            ValueWrapper::F64(v) => get_ptr_impl(v),
            ValueWrapper::Ptr(v) => get_ptr_impl(v),
            ValueWrapper::Size(v) => get_ptr_impl(v),
            ValueWrapper::Struct(v) => v.data.as_ptr() as *mut c_void,
            ValueWrapper::Closure(_) => {
                panic!("closure can only be used as argument directly, use it as pointer")
            }
        }
    }
}

struct FuncWrapper {
    func: String,
    args: Vec<TypeWrapper>,
    ret: TypeWrapper,
}

impl TypeWrapper {
    pub fn get_ffi_type(&self) -> Type {
        match self {
            TypeWrapper::Void => Type::void(),
            TypeWrapper::Bool => Type::u8(),
            TypeWrapper::Size => Type::isize(),
            TypeWrapper::U8 => Type::u8(),
            TypeWrapper::U16 => Type::u16(),
            TypeWrapper::U32 => Type::u32(),
            TypeWrapper::U64 => Type::u64(),
            TypeWrapper::I8 => Type::i8(),
            TypeWrapper::I16 => Type::i16(),
            TypeWrapper::I32 => Type::i32(),
            TypeWrapper::I64 => Type::i64(),
            TypeWrapper::F32 => Type::f32(),
            TypeWrapper::F64 => Type::f64(),
            TypeWrapper::Ptr => Type::pointer(),
            TypeWrapper::Struct(fields) => Type::structure(fields.iter().map(|t| t.get_ffi_type())),
        }
    }
}

fn value_from_buffer(type_wrapper: &TypeWrapper, buffer: &[u8]) -> ValueWrapper {
    match type_wrapper {
        TypeWrapper::Void => ValueWrapper::Void,
        TypeWrapper::Bool => ValueWrapper::Bool(buffer[0] != 0),
        TypeWrapper::U8 => ValueWrapper::U8(u8::from_ne_bytes(
            buffer[0..std::mem::size_of::<u8>()].try_into().unwrap(),
        )),
        TypeWrapper::U16 => ValueWrapper::U16(u16::from_ne_bytes(
            buffer[0..std::mem::size_of::<u16>()].try_into().unwrap(),
        )),
        TypeWrapper::U32 => ValueWrapper::U32(u32::from_ne_bytes(
            buffer[0..std::mem::size_of::<u32>()].try_into().unwrap(),
        )),
        TypeWrapper::U64 => ValueWrapper::U64(u64::from_ne_bytes(
            buffer[0..std::mem::size_of::<u64>()].try_into().unwrap(),
        )),
        TypeWrapper::I8 => ValueWrapper::I8(buffer[0] as i8),
        TypeWrapper::I16 => ValueWrapper::I16(i16::from_ne_bytes(
            buffer[0..std::mem::size_of::<i16>()].try_into().unwrap(),
        )),
        TypeWrapper::I32 => ValueWrapper::I32(i32::from_ne_bytes(
            buffer[0..std::mem::size_of::<i32>()].try_into().unwrap(),
        )),
        TypeWrapper::I64 => ValueWrapper::I64(i64::from_ne_bytes(
            buffer[0..std::mem::size_of::<i64>()].try_into().unwrap(),
        )),
        TypeWrapper::F32 => ValueWrapper::F32(f32::from_ne_bytes(
            buffer[0..std::mem::size_of::<f32>()].try_into().unwrap(),
        )),
        TypeWrapper::F64 => ValueWrapper::F64(f64::from_ne_bytes(
            buffer[0..std::mem::size_of::<f64>()].try_into().unwrap(),
        )),
        TypeWrapper::Ptr => ValueWrapper::Ptr(usize::from_ne_bytes(
            buffer[0..std::mem::size_of::<usize>()].try_into().unwrap(),
        )),
        TypeWrapper::Size => ValueWrapper::Size(usize::from_ne_bytes(
            buffer[0..std::mem::size_of::<usize>()].try_into().unwrap(),
        )),
        TypeWrapper::Struct(_) => ValueWrapper::Struct(StructBuffer {
            data: buffer.to_vec(),
        }),
    }
}

macro_rules! get_type_nif {
    ($i:ident, $t:expr) => {
        #[rustler::nif]
        fn $i() -> ResourceArc<TypeWrapper> {
            ResourceArc::new($t)
        }
    };
}

get_type_nif!(get_void_type, TypeWrapper::Void);
get_type_nif!(get_bool_type, TypeWrapper::Bool);
get_type_nif!(get_u8_type, TypeWrapper::U8);
get_type_nif!(get_u16_type, TypeWrapper::U16);
get_type_nif!(get_u32_type, TypeWrapper::U32);
get_type_nif!(get_u64_type, TypeWrapper::U64);
get_type_nif!(get_i8_type, TypeWrapper::I8);
get_type_nif!(get_i16_type, TypeWrapper::I16);
get_type_nif!(get_i32_type, TypeWrapper::I32);
get_type_nif!(get_i64_type, TypeWrapper::I64);
get_type_nif!(get_size_type, TypeWrapper::Size);
get_type_nif!(get_f32_type, TypeWrapper::F32);
get_type_nif!(get_f64_type, TypeWrapper::F64);
get_type_nif!(get_ptr_type, TypeWrapper::Ptr);

#[rustler::nif]
fn get_struct_type(iter: ListIterator) -> ResourceArc<TypeWrapper> {
    let fields: Result<Vec<TypeWrapper>, Error> = iter
        .map(|x| {
            (x.decode::<ResourceArc<TypeWrapper>>().map(|x| (*x).clone()))
                .map_err(|_| Error::BadArg)
        })
        .collect();
    ResourceArc::new(TypeWrapper::Struct(fields.unwrap()))
}

macro_rules! get_value_nif {
    ($i:ident, $value_t:tt, $wrapper_t:expr) => {
        #[rustler::nif]
        fn $i(v: $value_t) -> ResourceArc<ValueWrapper> {
            ResourceArc::new($wrapper_t(v))
        }
    };
}

get_value_nif!(get_u8_value, u8, ValueWrapper::U8);
get_value_nif!(get_u16_value, u16, ValueWrapper::U16);
get_value_nif!(get_u32_value, u32, ValueWrapper::U32);
get_value_nif!(get_u64_value, u64, ValueWrapper::U64);
get_value_nif!(get_i8_value, i8, ValueWrapper::I8);
get_value_nif!(get_i16_value, i16, ValueWrapper::I16);
get_value_nif!(get_i32_value, i32, ValueWrapper::I32);
get_value_nif!(get_i64_value, i64, ValueWrapper::I64);
get_value_nif!(get_f32_value, f32, ValueWrapper::F32);
get_value_nif!(get_f64_value, f64, ValueWrapper::F64);

// TODO: make appending null optional
#[rustler::nif]
fn get_c_string_value(v: &str) -> ResourceArc<ValueWrapper> {
    let mut buffer = StructBuffer::new(v.len());
    let _ = &buffer.data.write_all(v.as_bytes()).unwrap();
    let p: *const i32 = std::ptr::null();
    let null = p as u8;
    buffer.data.push(null);
    ResourceArc::new(ValueWrapper::Struct(buffer))
}

#[rustler::nif]
fn get_null_ptr_value() -> ResourceArc<ValueWrapper> {
    let p: *const i32 = std::ptr::null();
    let null = p as u8;
    ResourceArc::new(ValueWrapper::Ptr(null.into()))
}

#[derive(Clone)]
struct ClosureUserData {
    pub pid: LocalPid,
    pub callback_id: Atom,
    args: Vec<TypeWrapper>,
    ret: TypeWrapper,
    // maybe a the process id/env to make sure it is the same caller?
}

impl std::fmt::Debug for ClosureUserData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosureUserData")
            .field("args", &self.args)
            .field("ret", &self.ret)
            .finish()
    }
}

#[derive(Debug)]
struct ClosureHandleWrapper {
    pub ptr: *mut libffi::raw::ffi_closure,
}

impl Drop for ClosureHandleWrapper {
    fn drop(&mut self) {
        unsafe {
            libffi::low::closure_free(self.ptr);
        }
    }
}
#[derive(Debug)]
struct ClosureWrapper {
    cif: Box<Cif>, // must be alive when closure is called
    user_data: Box<ClosureUserData>,
    _handle: ClosureHandleWrapper,
    code: CodePtr,
}
unsafe impl Sync for ClosureWrapper {}
unsafe impl Send for ClosureWrapper {}

// A mutex and condition variable to guard Elixir-Rust interaction

pub struct LogicalToken {
    pub cvar: Condvar,
    pub result: Mutex<Option<bool>>,
}

// TODO: call closure_free when ClosureWrapper is dropped

mod atoms {
    rustler::atoms! { drop_closure}
}

unsafe extern "C" fn callback(
    cif: &libffi::raw::ffi_cif,
    result: &mut c_void,
    args: *const *const c_void,
    user_data: &ClosureUserData, // using static here to avoid lifetime issues
) {
    let callback_token = ResourceArc::new(LogicalToken {
        cvar: Condvar::new(),
        result: Mutex::new(None),
    });

    let mut msg_env = OwnedEnv::new();
    let ret_ptr_value = ResourceArc::new(ValueWrapper::Ptr(result as *mut c_void as usize));
    // collect arguments as ValueWrappers
    let arg_values: Vec<ResourceArc<ValueWrapper>> = (&user_data.args)
        .into_iter()
        .enumerate()
        .map(|(i, t)| {
            let arg_ptr = *args.add(i);
            let size = (*(*cif.arg_types.add(i))).size;
            let buffer = slice::from_raw_parts(arg_ptr as *const u8, size);
            let arg_val = value_from_buffer(&t, buffer);
            ResourceArc::new(arg_val)
        })
        .collect();
    // send callback handler process the message
    let user_data_copy = (*user_data).clone();
    let callback_token_copy = callback_token.clone();
    let handler = thread::spawn(move || {
        msg_env.send_and_clear(&user_data_copy.pid, |env| {
            (
                user_data_copy.callback_id,
                arg_values,
                ret_ptr_value,
                callback_token_copy,
            )
                .encode(env)
        });
    });

    handler.join().unwrap();
    let token = &(*callback_token);
    let mut logical_result = token.result.lock().unwrap();
    while !logical_result.is_some() {
        logical_result = token.cvar.wait(logical_result).unwrap();
    }
}

#[rustler::nif]
fn finish_callback(token: ResourceArc<LogicalToken>, success: bool) {
    let mut result = token.result.lock().unwrap();
    *result = Some(success);
    token.cvar.notify_one();
}

// TODO: option to use a worker thread to call send message in callback
// Get a closure pack args from C as a message and send it to callback handler process
#[rustler::nif]
fn get_closure(
    callback_handler: LocalPid,
    callback_id: Atom,
    ret: ResourceArc<TypeWrapper>,
    args: ListIterator, // for some reason, args has to be the last argument other wise the macro won't compile
) -> ResourceArc<ValueWrapper> {
    let arg_types: Vec<TypeWrapper> = args
        .map(|t| (*((t.decode::<ResourceArc<TypeWrapper>>()).unwrap())).clone())
        .collect();
    let args: Vec<Type> = arg_types.iter().map(|t| t.get_ffi_type()).collect();
    let cif = Builder::new().res(ret.get_ffi_type()).args(args).into_cif();
    let cif = Box::new(cif);
    let (closure, code) = libffi::low::closure_alloc();
    let mut wrapper = ClosureWrapper {
        cif,
        user_data: Box::new(ClosureUserData {
            pid: callback_handler,
            args: arg_types,
            ret: (*ret).clone(),
            callback_id,
        }),
        _handle: ClosureHandleWrapper { ptr: closure },
        code: code,
    };
    unsafe {
        libffi::low::prep_closure(
            closure,
            wrapper.cif.as_raw_ptr(),
            callback,
            &mut *wrapper.user_data,
            code,
        )
        .unwrap();
        ResourceArc::new(ValueWrapper::Closure(wrapper))
    }
}

// NOTE: return a Cif so that the caller owns it, returning the ffi type pointer is dangerous when Cif has been dropped
fn collect_struct_rtype_cif(types_iter: ListIterator) -> Cif {
    let types: Vec<ResourceArc<TypeWrapper>> = types_iter
        .map(|t| ((t.decode::<ResourceArc<TypeWrapper>>()).unwrap()))
        .collect();
    let elements: Vec<Type> = types.iter().map(|t| t.get_ffi_type()).collect();
    let struct_type = Type::structure(elements);
    let cif = Builder::new().res(struct_type).into_cif();
    cif
}

fn create_struct_rtype_cif(types: &Vec<TypeWrapper>) -> Cif {
    let elements: Vec<Type> = types.iter().map(|t| t.get_ffi_type()).collect();
    let struct_type = Type::structure(elements);
    let cif = Builder::new().res(struct_type).into_cif();
    cif
}

unsafe fn n_th_size_in_type(i: usize, t_ptr: *const ffi_type) -> usize {
    let n_element_type_ptr = (*t_ptr).elements.add(i);
    let count = (*(*n_element_type_ptr)).size;
    return count;
}

#[rustler::nif]
fn get_struct_value(types_iter: ListIterator, values: ListIterator) -> ResourceArc<ValueWrapper> {
    let cif = collect_struct_rtype_cif(types_iter);
    unsafe {
        // get sizes after applying the proper layout
        let rtype_ptr = (*cif.as_raw_ptr()).rtype as *const ffi_type;
        let total_size: usize = (*rtype_ptr).size;
        let mut buffer = StructBuffer::new(total_size);
        let mut dst_ptr = buffer.data.as_mut_ptr();
        values.enumerate().for_each(|(i, v)| {
            let value_ref = (v.decode::<ResourceArc<ValueWrapper>>()).unwrap();
            // how many bytes of a aligned field has
            let count = n_th_size_in_type(i, rtype_ptr);
            let src_ptr = value_ref.get_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, count);
            dst_ptr = dst_ptr.add(count);
        });
        buffer.data.set_len(total_size);
        ResourceArc::new(ValueWrapper::Struct(buffer))
    }
}

#[rustler::nif]
fn get_ptr(value: ResourceArc<ValueWrapper>) -> ResourceArc<ValueWrapper> {
    ResourceArc::new(ValueWrapper::Ptr(value.get_ptr() as usize))
}

#[rustler::nif]
fn as_ptr(value: ResourceArc<ValueWrapper>) -> ResourceArc<ValueWrapper> {
    match &*value {
        ValueWrapper::I32(v) => ResourceArc::new(ValueWrapper::Ptr(*v as usize)),
        ValueWrapper::I64(v) => ResourceArc::new(ValueWrapper::Ptr(*v as usize)),
        ValueWrapper::Ptr(v) => ResourceArc::new(ValueWrapper::Ptr(*v)),
        ValueWrapper::Closure(v) => {
            ResourceArc::new(ValueWrapper::Ptr(v.code.as_mut_ptr() as usize))
        }
        _ => panic!("Cannot convert to ptr, maybe what you want is get_ptr?"),
    }
}

#[rustler::nif]
fn as_binary(value: ResourceArc<ValueWrapper>) -> OwnedBinary {
    match &*value {
        // TODO: fix duplication
        ValueWrapper::Struct(buffer) => {
            let mut binary = OwnedBinary::new(buffer.data.len()).unwrap();
            binary.as_mut_slice().write_all(&buffer.data).unwrap();
            binary
        }
        v => {
            panic!("Unsupported value type to convert to binary {:?}", v);
        }
    }
}

// extract non-struct value directly
// struct value should be extracted with types, because it is used to mimic array as well
#[rustler::nif]
fn extract<'a>(env: Env<'a>, value: ResourceArc<ValueWrapper>) -> Term<'a> {
    EncoderWrapper { t: None, v: &value }.encode(env)
}

struct EncoderWrapper<'a> {
    t: Option<&'a TypeWrapper>, // required if it is struct type
    v: &'a ValueWrapper,
}

impl Encoder for EncoderWrapper<'_> {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        match (self.t, self.v) {
            (_, ValueWrapper::Bool(v)) => v.encode(env),
            (_, ValueWrapper::F64(v)) => v.encode(env),
            (_, ValueWrapper::F32(v)) => v.encode(env),
            (_, ValueWrapper::I8(v)) => v.encode(env),
            (_, ValueWrapper::I16(v)) => v.encode(env),
            (_, ValueWrapper::I32(v)) => v.encode(env),
            (_, ValueWrapper::I64(v)) => v.encode(env),
            (_, ValueWrapper::U8(v)) => v.encode(env),
            (_, ValueWrapper::U16(v)) => v.encode(env),
            (_, ValueWrapper::U32(v)) => v.encode(env),
            (_, ValueWrapper::U64(v)) => v.encode(env),
            (_, ValueWrapper::Size(v)) => v.encode(env),
            (_, ValueWrapper::Ptr(v)) => v.encode(env),
            (Some(TypeWrapper::Struct(fields)), ValueWrapper::Struct(buffer)) => unsafe {
                let cif = create_struct_rtype_cif(fields);
                let rtype_ptr = (*cif.as_raw_ptr()).rtype as *const ffi_type;
                let mut offset: usize = 0;
                let terms: Vec<Term> = fields
                    .into_iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let size = n_th_size_in_type(i, rtype_ptr);
                        let end = offset + size;
                        let v = value_from_buffer(f, &buffer.data[offset..end]);
                        offset = end;
                        EncoderWrapper { t: Some(&f), v: &v }.encode(env)
                    })
                    .collect();
                terms.encode(env)
            },
            _ => panic!("Unsupported value to encode {:?}", self.v),
        }
    }
}

#[rustler::nif]
fn extract_struct<'a>(
    env: Env<'a>,
    struct_type_wrapper: ResourceArc<TypeWrapper>,
    value: ResourceArc<ValueWrapper>,
) -> Term<'a> {
    // TODO: fix duplication
    let buffer: &[u8];
    match &*value {
        ValueWrapper::Struct(v) => buffer = v.data.as_slice(),
        v => {
            panic!("not a struct to extract, value: {:?}", v);
        }
    };
    EncoderWrapper {
        t: Some(&struct_type_wrapper),
        v: &ValueWrapper::Struct(StructBuffer {
            data: buffer.to_vec(),
        }),
    }
    .encode(env)
}

#[rustler::nif]
fn extract_c_string_as_binary_string(
    ptr: ResourceArc<ValueWrapper>,
    length: ResourceArc<ValueWrapper>,
) -> String {
    let length: usize = match *length {
        ValueWrapper::I32(v) => v.try_into().unwrap(),
        ValueWrapper::U8(v) => v.try_into().unwrap(),
        ValueWrapper::U16(v) => v.try_into().unwrap(),
        ValueWrapper::U32(v) => v.try_into().unwrap(),
        ValueWrapper::U64(v) => v.try_into().unwrap(),
        ValueWrapper::I8(v) => v.try_into().unwrap(),
        ValueWrapper::I16(v) => v.try_into().unwrap(),
        ValueWrapper::I64(v) => v.try_into().unwrap(),
        ValueWrapper::Size(v) => v.try_into().unwrap(),
        _ => panic!("not a integer, value: {:?}", *length),
    };
    let ptr = match *ptr {
        ValueWrapper::Ptr(v) => v as *mut u8,
        _ => panic!("not a ptr, value: {:?}", *ptr),
    };
    unsafe {
        let mut read = slice::from_raw_parts(ptr, length);
        let mut buf = String::new();
        read.read_to_string(&mut buf).unwrap();
        buf
    }
}

fn get_buffer_from_struct_wrapper<'a>(value: &'a ValueWrapper) -> &'a [u8] {
    // TODO: fix duplication
    let buffer: &[u8];
    match &*value {
        ValueWrapper::Struct(v) => buffer = v.data.as_slice(),
        v => {
            panic!("not a struct to extract: {:?}", v);
        }
    };
    return buffer;
}

#[rustler::nif]
fn access_struct_field_as_value<'a>(
    struct_type_wrapper: ResourceArc<TypeWrapper>,
    value: ResourceArc<ValueWrapper>,
    index: usize,
) -> ResourceArc<ValueWrapper> {
    if let TypeWrapper::Struct(fields) = &*struct_type_wrapper {
        unsafe {
            let buffer = get_buffer_from_struct_wrapper(&value).to_vec();
            let cif = create_struct_rtype_cif(&fields);
            let rtype_ptr = (*cif.as_raw_ptr()).rtype as *const ffi_type;
            // sum the sizes of previous fields
            let mut start_offset = 0;
            for i in 0..index {
                let size = n_th_size_in_type(i, rtype_ptr);
                start_offset += size;
            }
            // get the size of the field and get where this field ends
            let end_offset = start_offset + n_th_size_in_type(index, rtype_ptr);
            let v = value_from_buffer(&fields[index], &buffer[start_offset..end_offset]);
            ResourceArc::new(v)
        }
    } else {
        panic!("not a struct to access field {:?}", *struct_type_wrapper);
    }
}

#[rustler::nif]
fn get_func(
    func: Term,
    rtype: ResourceArc<TypeWrapper>,
    iter: ListIterator,
) -> ResourceArc<FuncWrapper> {
    // TODO: extract function
    let fields: Result<Vec<TypeWrapper>, Error> = iter
        .map(|x| {
            (x.decode::<ResourceArc<TypeWrapper>>().map(|x| (*x).clone()))
                .map_err(|_| Error::BadArg)
        })
        .collect();
    ResourceArc::new(FuncWrapper {
        func: func.atom_to_string().unwrap(),
        args: fields.unwrap(),
        ret: (*rtype).clone(),
    })
}

struct LibWrapper {
    pub lib: Library,
}

impl LibWrapper {
    pub fn new(lib: &str) -> NifResult<Self> {
        unsafe {
            Library::new(lib)
                .and_then(|lib| Ok(LibWrapper { lib }))
                .map_err(|e| Error::Term(Box::new(format!("{}", e))))
        }
    }
}

// TODO: use macro to create a call_cif for each kind of return type
pub fn get_type(type_name: Term) -> Type {
    if type_name.atom_to_string().unwrap() == "" {
        Type::void()
    } else {
        Type::void()
    }
}

#[rustler::nif]
fn get_lib(lib: &str) -> NifResult<ResourceArc<LibWrapper>> {
    return LibWrapper::new(lib).map(|x| ResourceArc::new(x));
}

// have to use dirty scheduler to prevent deadlock when callback is running on the current normal scheduler
#[rustler::nif(schedule = "DirtyCpu")]
fn call_func(
    lib_wrapper: ResourceArc<LibWrapper>,
    func_wrapper: ResourceArc<FuncWrapper>,
    iter: ListIterator, /* list of ValueWrapper */
) -> ResourceArc<ValueWrapper> {
    unsafe {
        let lib = &lib_wrapper.lib;
        let func: Symbol<*mut c_void> = lib.get(func_wrapper.func.as_bytes()).unwrap();
        let args: Vec<Type> = func_wrapper.args.iter().map(|t| t.get_ffi_type()).collect();
        let cif = Builder::new()
            .res(func_wrapper.ret.get_ffi_type())
            .args(args)
            .into_cif();
        let arg_values: Vec<ArgWrapper> = iter
            .map(|x| (*x.decode::<ResourceArc<ValueWrapper>>().unwrap()).get_arg())
            .collect();
        let rsize = (*(*cif.as_raw_ptr()).rtype).size;
        let mut ret_buffer: Vec<u8> = Vec::with_capacity(rsize);
        ret_buffer.set_len(rsize);
        let fun = CodePtr(*func);
        libffi::raw::ffi_call(
            cif.as_raw_ptr(),
            Some(*fun.as_safe_fun()),
            ret_buffer.as_mut_ptr() as *mut c_void,
            arg_values.as_ptr() as *mut *mut c_void,
        );
        ResourceArc::new(value_from_buffer(&func_wrapper.ret, &ret_buffer))
    }
}

pub fn on_load(env: Env) -> bool {
    rustler::resource!(StructBuffer, env);
    rustler::resource!(TypeWrapper, env);
    rustler::resource!(FuncWrapper, env);
    rustler::resource!(ValueWrapper, env);
    rustler::resource!(LibWrapper, env);
    rustler::resource!(LogicalToken, env);
    rustler::resource!(ClosureWrapper, env);
    true
}

fn load(env: rustler::Env, _: rustler::Term) -> bool {
    on_load(env);
    true
}

rustler::init!(
    "Elixir.Exotic.NIF",
    [
        get_lib,
        get_size_type,
        get_bool_type,
        get_u8_type,
        get_u16_type,
        get_u32_type,
        get_u64_type,
        get_i8_type,
        get_i16_type,
        get_i32_type,
        get_i64_type,
        get_f32_type,
        get_f64_type,
        get_ptr_type,
        get_void_type,
        get_struct_type,
        get_u8_value,
        get_u16_value,
        get_u32_value,
        get_u64_value,
        get_i8_value,
        get_i16_value,
        get_i32_value,
        get_i64_value,
        get_f32_value,
        get_f64_value,
        get_c_string_value,
        get_null_ptr_value,
        get_closure,
        get_struct_value,
        extract,
        extract_struct,
        extract_c_string_as_binary_string,
        access_struct_field_as_value,
        get_ptr,
        as_ptr,
        as_binary,
        get_func,
        call_func,
        finish_callback,
    ],
    load = load
);
