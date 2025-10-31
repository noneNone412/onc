use onc::controller::controller::Controller;

fn main() {
    let controller_ = Controller::new();
    controller_.run();
    /* println!("\n\n\n ---------------------");
       let cargo_toml_dir = env!("CARGO_MANIFEST_DIR");
       let cargo_toml_path = format!("{}/Cargo.toml", cargo_toml_dir);
       println!("Path to Cargo.toml: {}", cargo_toml_path);
    */
    /* database config later
    println!("{}", master::database_path());
    let dbConnect = duckDB_system::DuckDbSystem::new("mydb");
    dbConnect.connect();
    */
    //write_shaders::write(String::from("x0a"), String::from("(a+b+c)"));
    // let exec1 = ShaderExec::new("shader1.spv".to_string(), vec![], Arc::clone(&shared_state));
    // let exec2 = ShaderExec::new("shader2.spv".to_string(), vec![], Arc::clone(&shared_state));
    /*
    let state = saved_state::SavedState::new();
    let shared_state = Arc::new(state);
    let v2: shaders_exec::ShaderExec = shaders_exec::ShaderExec::new(
        String::from("x0a"),
        Vec::<Vec<f64>>::new(),
        Arc::clone(&shared_state),
    );
    v2.exec_shader();

    println!("\n\n\n ---------------------"); */
    /*
    check if is running first time
    create the config file in the system
    create the databases
    */

    println!("Code exited successfully");
}
