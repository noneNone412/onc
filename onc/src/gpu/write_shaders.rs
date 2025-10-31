use regex::Regex;
use serde_json::json;
use serde_json::{to_writer_pretty, Map, Result, Value};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn resevedGLSL_functions() -> Vec<String> {
    return vec![String::from("log"), String::from("abs")];
}

pub fn write(name: String, equation: String) {
    let glslcPath: &Path = Path::new("/home/none/sdks/vulkan 1.4/x86_64/bin/glslc");
    // now change the equation
    // convert input_eqn to shader_eqn
    // remove braces first
    let mut inputString = equation.replace('(', " ");
    inputString = inputString.replace(')', " ");
    // remove basic operators next
    inputString = inputString.replace('+', " ");
    inputString = inputString.replace('-', " ");
    inputString = inputString.replace('*', " ");
    inputString = inputString.replace('/', " ");
    inputString = inputString.replace('%', " ");
    println!("input string: {}", inputString);

    // convert  string to vector
    let mut inputVec: Vec<String> = inputString
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    // Print the vector
    println!("{:?}", &inputVec);
    // remove numbers from the vector
    inputVec = inputVec
        .into_iter()
        .filter(|s| s.parse::<f64>().is_err())
        .collect();

    println!("once again{:?}", &inputVec);
    // remove redundancy from vector
    /* let mut unique_vec: Vec<String> = inputVec
    .into_iter()
    .collect::<HashSet<String>>()
    .into_iter()
    .collect(); */
    let mut unique_vec: Vec<String> = Vec::new();
    for ss in &inputVec {
        if unique_vec.contains(ss) {
            continue;
        }
        unique_vec.push(ss.to_string());
    }
    println!("once again 2 {:?}", &unique_vec);
    // remove other_functions
    unique_vec.retain(|item| !resevedGLSL_functions().contains(item));
    println!("unique vec{:?}", &unique_vec);
    // build final shader_eqn
    let mut shader_eqn = equation.to_string();

    for var in &unique_vec {
        // Create a regex pattern to match the whole word (similar to "\\b" in C++)
        let pattern = format!(r"\b{}\b", var);
        let re = Regex::new(&pattern).unwrap();
        // Replace occurrences of the variable with the transformed variable
        shader_eqn = re
            .replace_all(&shader_eqn, format!("{}[reservedX]", var))
            .to_string();
    }
    println!("shader_eqn{:?}", &shader_eqn);
    // create a file in named shader name to a specified location
    let project_root: &str = env!("CARGO_MANIFEST_DIR");
    let file_path = format!("{}/src/gpu/shaders/readable/", project_root);
    let file_path_full: String = file_path.clone() + &name + ".comp";
    let compiled_path: String = format!("{}/src/gpu/shaders/compiled/", project_root);
    if fs::metadata(file_path_full.clone()).is_ok() {
        // File exists, attempt to delete it
        match fs::remove_file(file_path_full.clone()) {
            Ok(_) => println!("File deleted successfully"),
            Err(e) => eprintln!("Failed to delete file: {}", e),
        }
    } else {
        println!("File does not exist.");
    }
    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .write(true)
        .open(&file_path_full)
    {
        Ok(file) => {
            println!("File opened Successfully");
            file
        }
        Err(e) => {
            eprintln!("Error opening file: {}", e);
            return;
        }
    };

    //create the json helper file
    let json_path_full = format!("{}/src/gpu/shaders/json/{}.json", project_root, &name);
    if fs::metadata(json_path_full.clone()).is_ok() {
        // File exists, attempt to delete it
        match fs::remove_file(json_path_full.clone()) {
            Ok(_) => println!("File deleted successfully"),
            Err(e) => eprintln!("Failed to delete file: {}", e),
        }
    } else {
        println!("File does not exist.");
    }
    let mut json_file = match OpenOptions::new()
        .create(true)
        .append(true)
        .write(true)
        .open(&json_path_full)
    {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error opening file: {}", e);
            return;
        }
    };

    let mut shaderCode: String = String::from("#version 450");
    shaderCode.push_str("\n");
    shaderCode.push_str("#extension GL_ARB_gpu_shader_fp64 : enable");

    shaderCode.push_str("\n\n");

    shaderCode.push_str("layout(constant_id = 0) const uint WORKGROUP_SIZE_X = 1;");
    shaderCode.push_str("\n\n");

    shaderCode.push_str("layout(");
    shaderCode.push_str("\n");
    shaderCode.push_str("\t");
    shaderCode.push_str("local_size_x_id = 0");
    shaderCode.push_str("\n");
    shaderCode.push_str(") in;");
    shaderCode.push_str("\n\n");

    let mut root = Map::new();
    let mut in_vec: Vec<Value> = Vec::new();
    // input descriptors set
    for i in 0..unique_vec.len() {
        shaderCode.push_str("layout(set = 0, binding = ");
        let s = i.to_string(); // Convert integer to String
        let str_slice: &str = &s;
        shaderCode.push_str(str_slice);
        shaderCode.push_str(") readonly buffer input_");
        let str_slice2: &str = &unique_vec[i];
        shaderCode.push_str(&str_slice2);
        shaderCode.push_str(" {");
        shaderCode.push_str("\n");
        shaderCode.push_str("\t");
        shaderCode.push_str("double ");
        shaderCode.push_str(&str_slice2);
        shaderCode.push_str("[];");
        shaderCode.push_str("\n");
        shaderCode.push_str("};");
        shaderCode.push_str("\n\n");

        // add to json helper file
        let a_objects = json!({ "set": 0 ,"binding": i});
        in_vec.push(a_objects);
    }
    root.insert("inputDescriptor".to_string(), Value::Array(in_vec));
    // output descriptor sets

    shaderCode.push_str(
        "layout(set = 1, binding = 0) writeonly buffer OutputData { double outData[]; };",
    );
    let a_objects = vec![json!({ "set":1 ,"binding": 0 })];
    root.insert("outputDescriptor".to_string(), Value::Array(a_objects));

    shaderCode.push_str("\n");
    shaderCode.push_str("layout(push_constant) uniform PushConstants {\n");

    shaderCode.push_str("\t int xCount; ");
    shaderCode.push_str("\n");
    shaderCode.push_str("} params;");
    shaderCode.push_str("\n");
    shaderCode.push_str("void main() {\n");
    shaderCode.push_str("\t uint reservedX = gl_GlobalInvocationID.x; \n");
    shaderCode.push_str("\t if (reservedX >= uint(params.xCount)) {\n");
    shaderCode.push_str("\t return;\n");
    shaderCode.push_str("\t }\n");
    shaderCode.push_str("\t outData[reservedX] = ");
    let str_shaderEqn: &str = &shader_eqn;
    shaderCode.push_str(str_shaderEqn);
    shaderCode.push_str(";");
    shaderCode.push_str("\n}");

    if let Err(e) = file.write_all(shaderCode.as_bytes()) {
        eprintln!("Error writing to file: {}", e);
        return;
    }
    // write to json file
    let writer = BufWriter::new(json_file);
    let jdata = Value::Object(root);
    to_writer_pretty(writer, &jdata).expect("Failed to write JSON");
    //  close file as it is still in  scope
    drop(file);

    // compile shader
    let output_file = format!("{}/src/gpu/shaders/compiled/{}.spv", project_root, &name);
    let status = Command::new(&glslcPath)
        .arg(&file_path_full)
        .arg("-o")
        .arg(&output_file)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("Shader compiled successfully to {}", output_file);
        }
        _ => {
            eprintln!("Failed to compile shader");
        }
    }
}
