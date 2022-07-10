use std::{path::Path, process::Command};

use cmake;
use glob::glob;
use std::env;
use which::which;

// llvm config code from: https://github.com/femtomc/mlir-sys
fn llvm_config_shim(arg: &str) -> String {
    let cmd_path = match std::env::var("LLVM_CONFIG_PATH") {
        Ok(v) => std::path::PathBuf::from(v),
        Err(_e) => which("llvm-config")
            .or(which("llvm-config-15"))
            .or(which("llvm-config-14"))
            .unwrap(),
    };
    let call = format!("{} {}", cmd_path.to_str().unwrap(), arg);
    let tg = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(["/C", &call[..]])
            .output()
            .expect("failed to execute process")
    } else {
        Command::new("sh")
            .arg("-c")
            .arg(&call[..])
            .output()
            .expect("failed to execute process")
    }
    .stdout;
    let mut s = String::from_utf8_lossy(&tg);
    s.to_mut().pop();
    return s.to_string();
}

fn main() {
    // Builds the project in the directory located in `libfoo`, installing it
    // into $OUT_DIR

    // This follows the practices in rust-bindgen
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-env-changed=LIBCLANG_STATIC_PATH");

    for entry in glob("met/**/CMakeLists.txt").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => println!("cargo:rerun-if-changed={}", path.display()),
            Err(e) => println!("{:?}", e),
        }
    }
    for entry in glob("met/**/*.cpp").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => println!("cargo:rerun-if-changed={}", path.display()),
            Err(e) => println!("{:?}", e),
        }
    }
    for entry in glob("met/**/*.h").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => println!("cargo:rerun-if-changed={}", path.display()),
            Err(e) => println!("{:?}", e),
        }
    }
    println!("cargo:rerun-if-changed=wrapper.h");
    let libdir = llvm_config_shim("--libdir");
    for entry in glob(Path::new(&libdir).join("**/*").to_str().unwrap())
        .expect("Failed to read glob pattern")
    {
        match entry {
            Ok(path) => println!("cargo:rerun-if-changed={}", path.display()),
            Err(e) => println!("{:?}", e),
        }
    }
    let llvm_cmake_dir = Path::new(&libdir).join("cmake").join("llvm");
    let mlir_cmake_dir = Path::new(&libdir).join("cmake").join("mlir");
    let dst = cmake::Config::new("met")
        .define("LLVM_DIR", llvm_cmake_dir)
        .define("MLIR_DIR", mlir_cmake_dir)
        .target("install")
        .build();

    let lib_dir = Path::new(&dst.display().to_string()).join("lib");
    bindgen::builder()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", llvm_config_shim("--includedir")))
        .clang_arg(format!("-I{}", "met/include"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .unwrap()
        .write_to_file(Path::new(&env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .unwrap();
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=met");
    // println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN");
    // println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
