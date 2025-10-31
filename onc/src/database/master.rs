pub const fn database_path() -> &'static str {
    return concat!(env!("CARGO_MANIFEST_DIR"), "/data");
}
