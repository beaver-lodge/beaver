use std::fs;
use std::str::FromStr;

use proc_macro2::TokenStream;
use rustler::{
    Atom, Encoder, Env, NifRecord, NifResult, NifStruct, NifTuple, NifUnitEnum, NifUntaggedEnum,
    Term,
};

use syn::Type::Ptr;
use syn::__private::quote::quote;
use syn::__private::ToTokens;
use syn::{File, FnArg, ForeignItem, ForeignItemFn, Item, ItemType, PatType, TypeArray};
use syn::{ItemStruct, TypePath};
use syn::{Path, PathSegment};

mod atoms {
    rustler::atoms! { void, i, u, f, array, bool }
}

// This follows type convention in NX
#[derive(NifTuple, Debug)]
pub struct SizedType {
    name: Atom,
    size: i32,
}

#[derive(NifRecord, Debug)]
#[tag = "type_def"]
pub struct TypeDefReference {
    name: Atom,
}

#[derive(NifRecord, Debug)]
#[tag = "struct"]
pub struct StructTypeOfFields {
    fields: Vec<Type>,
}

#[derive(NifRecord, Debug)]
#[tag = "function"]

// head is the return type
pub struct FunctionType {
    types: Vec<Type>,
}

#[derive(NifRecord, Debug)]
#[tag = "ptr"]
pub struct PtrType {
    ty: Vec<Type>, // :void or the struct type
}

// Sizes' types very on different platforms
#[derive(NifUnitEnum, Debug)]
pub enum NamedType {
    Void,
    Isize,
    Usize,
    Char,
    Bool,
    Size,
}

#[derive(NifUntaggedEnum, Debug)]
pub enum Type {
    Sized(SizedType),
    Struct(TypeDefReference),
    StructAnonymous(StructTypeOfFields),
    FunctionAnonymous(FunctionType),
    Ptr(PtrType),
    Named(NamedType),
}

#[derive(NifStruct, Debug)]
#[module = "Exotic.CodeGen.Function"]
struct FunctionDef {
    name: Atom,
    args: Vec<(Atom, Type)>,
    ret: Type,
}

#[derive(NifStruct, Debug)]
#[module = "Exotic.CodeGen.Struct"]
pub struct StructDef {
    name: Atom,
    fields: Vec<(Atom, Type)>,
}

#[derive(NifStruct, Debug)]
#[module = "Exotic.CodeGen.TypeDef"]
pub struct TypeDef {
    name: Atom,
    ty: Type,
}

#[derive(NifStruct, Debug)]
#[module = "Exotic.Header"]
pub struct Header {
    file: String,
    module_name: Atom,
    search_paths: Vec<String>,
    functions: Vec<FunctionDef>,
    structs: Vec<StructDef>,
    type_defs: Vec<TypeDef>,
}

impl Header {
    fn get_sub_module_name<'a>(&self, env: Env<'a>, sub_module: &String) -> Atom {
        let mut capitalized: Vec<char> = sub_module.chars().collect();
        capitalized[0] = capitalized[0].to_uppercase().nth(0).unwrap();
        let module_names: Vec<String> = vec![
            self.module_name.to_term(env).atom_to_string().unwrap(),
            capitalized.into_iter().collect(),
        ];
        let module_name = module_names.join(".");
        Atom::from_str(env, &module_name).unwrap()
    }
}

fn get_void_ret_type() -> Type {
    Type::Named(NamedType::Void)
}

// TODO: accumulate all structs and function pointer types collected
fn get_type<'a>(env: Env<'a>, ty: &syn::Type, header: &Header) -> Type {
    match ty {
        Ptr(_) => Type::Ptr(PtrType {
            ty: vec![Type::Named(NamedType::Void)],
        }),
        syn::Type::Path(TypePath {
            qself: None,
            path: Path {
                leading_colon: _,
                segments,
            },
        }) if (segments.len() == 1) => match segments.first() {
            Some(PathSegment { ident, .. }) if ident.eq("isize") => Type::Named(NamedType::Isize),
            Some(PathSegment { ident, .. }) if ident.eq("i64") => Type::Sized(SizedType {
                name: atoms::i(),
                size: 64,
            }),
            Some(PathSegment { ident, .. }) if ident.eq("i32") => Type::Sized(SizedType {
                name: atoms::i(),
                size: 32,
            }),
            Some(PathSegment { ident, .. }) if ident.eq("i16") => Type::Sized(SizedType {
                name: atoms::i(),
                size: 16,
            }),
            Some(PathSegment { ident, .. }) if ident.eq("i8") => Type::Sized(SizedType {
                name: atoms::i(),
                size: 8,
            }),
            Some(PathSegment { ident, .. }) if ident.eq("u64") => Type::Sized(SizedType {
                name: atoms::u(),
                size: 64,
            }),
            Some(PathSegment { ident, .. }) if ident.eq("u32") => Type::Sized(SizedType {
                name: atoms::u(),
                size: 32,
            }),
            Some(PathSegment { ident, .. }) if ident.eq("u16") => Type::Sized(SizedType {
                name: atoms::u(),
                size: 16,
            }),
            Some(PathSegment { ident, .. }) if ident.eq("u8") => Type::Sized(SizedType {
                name: atoms::u(),
                size: 8,
            }),
            Some(PathSegment { ident, .. }) if ident.eq("f64") => Type::Sized(SizedType {
                name: atoms::f(),
                size: 64,
            }),
            Some(PathSegment { ident, .. }) if ident.eq("f32") => Type::Sized(SizedType {
                name: atoms::f(),
                size: 32,
            }),
            Some(PathSegment { ident, .. }) if ident.eq("size_t") => Type::Named(NamedType::Size),
            Some(PathSegment { ident, .. }) if ident.eq("bool") => Type::Named(NamedType::Bool),
            Some(PathSegment { ident, .. }) => Type::Struct(TypeDefReference {
                name: header.get_sub_module_name(env, &ident.to_string()),
            }),
            None => unreachable!(),
        },
        syn::Type::Path(TypePath {
            qself: None,
            path: Path {
                leading_colon: _,
                segments,
            },
        }) if (&segments[2].ident.to_string() == "Option") => {
            let seg = &segments[2];
            if let syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
                colon2_token: _,
                lt_token: _,
                args,
                gt_token: _,
            }) = &seg.arguments
            {
                assert!(args.len() == 1);
                let bracket_arg = &args[0];
                if let syn::GenericArgument::Type(syn::Type::BareFn(syn::TypeBareFn {
                    lifetimes: _,
                    unsafety: _,
                    abi: _,
                    fn_token: _,
                    paren_token: _,
                    inputs,
                    variadic: _,
                    output,
                })) = bracket_arg
                {
                    let args: Vec<Type> = inputs
                        .iter()
                        .map(|arg| get_type(env, &arg.ty, header))
                        .collect();
                    let ret = match output {
                        syn::ReturnType::Default => Type::Named(NamedType::Void),
                        syn::ReturnType::Type(_, ret_ty) => get_type(env, &ret_ty, header),
                    };
                    let types: Vec<Type> = vec![ret].into_iter().chain(args.into_iter()).collect();
                    let f = Type::FunctionAnonymous(FunctionType { types: types });
                    Type::Ptr(PtrType { ty: vec![f] })
                } else {
                    panic!("unexpected type in bracket: {:#?}", bracket_arg);
                }
            } else {
                panic!("unexpected type in option to be a function: {:#?}", seg);
            }
        }
        syn::Type::Array(TypeArray {
            bracket_token: _,
            elem,
            semi_token: _,
            // len: len,
            len:
                syn::Expr::Lit(syn::ExprLit {
                    attrs: _,
                    lit: syn::Lit::Int(len),
                }),
        }) => {
            let size = len.base10_parse::<usize>().unwrap();
            let mut fields = vec![];
            for _ in 0..size {
                let t = get_type(env, &elem, &header);
                // can't clone t so have to do this iteration
                fields.push(t);
            }
            Type::StructAnonymous(StructTypeOfFields { fields })
        }
        t => {
            // TODO: maybe a more idiomatic way to do this?
            if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_int)).unwrap()) {
                Type::Sized(SizedType {
                    name: atoms::i(),
                    size: 32,
                })
            } else if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_long)).unwrap()) {
                Type::Sized(SizedType {
                    name: atoms::i(),
                    size: 64,
                })
            } else if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_char)).unwrap()) {
                Type::Named(NamedType::Char)
            } else if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_schar)).unwrap()) {
                Type::Named(NamedType::Char)
            } else if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_uchar)).unwrap()) {
                Type::Named(NamedType::Char)
            } else if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_short)).unwrap()) {
                Type::Sized(SizedType {
                    name: atoms::i(),
                    size: 16,
                })
            } else if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_ushort)).unwrap()) {
                Type::Sized(SizedType {
                    name: atoms::u(),
                    size: 16,
                })
            } else if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_longlong)).unwrap()) {
                Type::Sized(SizedType {
                    name: atoms::i(),
                    size: 64,
                })
            } else if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_ulonglong)).unwrap())
            {
                Type::Sized(SizedType {
                    name: atoms::u(),
                    size: 64,
                })
            } else if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_ulong)).unwrap()) {
                Type::Sized(SizedType {
                    name: atoms::u(),
                    size: 32,
                })
            } else if t.eq(&syn::parse2::<syn::Type>(quote!(::std::os::raw::c_uint)).unwrap()) {
                Type::Sized(SizedType {
                    name: atoms::u(),
                    size: 32,
                })
            } else {
                panic!(
                    "Unsupported type: {:?} \n{:#?}",
                    t.to_token_stream().to_string(),
                    t
                );
            }
        }
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
// #[rustler::nif(schedule = "DirtyIo")]
fn parse_header<'a>(env: Env<'a>, header: Header) -> NifResult<Term<'a>> {
    let b = bindgen::builder().header(&header.file);
    let bindings = &header
        .search_paths
        .iter()
        .fold(b, |b, include_dir| {
            b.clang_arg(format!("-I/{}", include_dir))
        })
        .generate()
        .expect("Unable to generate bindings.");
    let mut buf = Vec::new();
    let writer = Box::new(&mut buf);
    bindings.write(writer).unwrap();
    bindings
        .write_to_file(format!("{}.rs", &header.file))
        .unwrap();
    let output = std::str::from_utf8(buf.as_slice()).unwrap().to_string();
    let token_stream = TokenStream::from_str(&output).unwrap();
    let file = syn::parse2::<File>(token_stream.clone()).unwrap();
    let token_str = format!("{:#?}", &file);
    fs::write(format!("{}.ast.rs", &header.file), token_str).expect("Unable to write file");
    let mut functions: Vec<FunctionDef> = vec![];
    let mut structs: Vec<StructDef> = vec![];
    let mut type_defs: Vec<TypeDef> = vec![];
    for item in file.items.into_iter() {
        match item {
            Item::ForeignMod(f) => {
                for i in f.items {
                    match i {
                        ForeignItem::Fn(ForeignItemFn {
                            sig,
                            attrs: _,
                            vis: _,
                            semi_token: _,
                        }) => {
                            let mut args: Vec<(Atom, Type)> = vec![];
                            for input in sig.inputs {
                                match input {
                                    FnArg::Typed(PatType {
                                        attrs: _,
                                        pat,
                                        colon_token: _,
                                        ty,
                                    }) => match *pat {
                                        // TODO: now this assumes every func arg has a name
                                        syn::Pat::Ident(i) => {
                                            let name =
                                                Atom::from_str(env, &i.ident.to_string()).unwrap();
                                            args.push((name, get_type(env, &ty, &header)));
                                        }
                                        _ => todo!(),
                                    },
                                    FnArg::Receiver(_) => {
                                        panic!("FFI functions cannot have receivers (self)")
                                    }
                                }
                            }
                            let ret = match sig.output {
                                syn::ReturnType::Type(_, ty) => get_type(env, &ty, &header),
                                syn::ReturnType::Default => get_void_ret_type(),
                            };
                            let function = FunctionDef {
                                name: Atom::from_str(env, &sig.ident.to_string()).unwrap(),
                                args,
                                ret,
                            };
                            let _ = &functions.push(function);
                        }
                        _ => (),
                    }
                }
            }
            Item::Struct(ItemStruct {
                attrs: _,
                vis: _,
                struct_token: _,
                ident,
                generics: _,
                fields,
                semi_token: _,
            }) => {
                let _ = &structs.push(StructDef {
                    name: header.get_sub_module_name(env, &ident.to_string()),
                    fields: fields
                        .into_iter()
                        .map(|f| {
                            let n = Atom::from_str(env, &f.ident.unwrap().to_string()).unwrap();
                            let t = get_type(env, &f.ty, &header);
                            (n, t)
                        })
                        .collect(),
                });
            }
            Item::Type(ItemType {
                attrs: _,
                vis: _,
                type_token: _,
                ident,
                generics: _,
                eq_token: _,
                ty,
                semi_token: _,
            }) => {
                let _ = &type_defs.push(TypeDef {
                    name: header.get_sub_module_name(env, &ident.to_string()),
                    ty: get_type(env, &ty, &header),
                });
            }
            _ => log::debug!("Ignore item:\n{:#?}", &item),
        }
    }
    // writeln!(&mut lock, "{}", output);
    NifResult::Ok(
        Header {
            file: header.file,
            module_name: header.module_name,
            search_paths: header.search_paths,
            functions,
            structs,
            type_defs,
        }
        .encode(env),
    )
}

rustler::init!("Elixir.Exotic.CodeGen", [parse_header]);
