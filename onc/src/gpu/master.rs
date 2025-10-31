pub const fn compiled_shader_path() -> &'static str {
    return concat!(env!("CARGO_MANIFEST_DIR"), "/data");
}
pub const fn readable_shader_path() -> &'static str {
    return concat!(env!("CARGO_MANIFEST_DIR"), "/data");
}
