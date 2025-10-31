// b:: writing only binary data in the file
/*
fn string_to_binary(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}
fn binary_to_string(bytes: &[u8]) -> Result<String, std::string::FromUtf8Error> {
    String::from_utf8(bytes.to_vec())
}
*/
// e:: writing only binary data in the file

// b:: reading only text data in the file
// will consume the memory of the line in the data
/*
use std::fs::File;
use std::io::{self, BufRead, BufReader};
fn main() -> io::Result<()> {
    let file = File::open("large_utf8_text.txt")?;
    let reader = BufReader::new(file);
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        println!("Line {}: {}", i + 1, line);
        // If you're worried about RAM, do NOT push to a Vec or store the line.
    }
    Ok(())
}
*/
// e:: reading only text data in the file

/*
pub struct FileSystem<'a> {
    database_name: &'a str,
}

impl<'a> FileSystem<'a> {
    pub fn new(database_name: &'a str) -> Self {
        FileSystem {
            database_name: (database_name),
        }
    }
    pub fn create_database(&self) {}
    pub fn create_table(&self) {}
    pub fn execute(&self) {}
}
 */
