// only to read from the python writes
// only for linux system
/*
use shared_memory::*;
use std::ffi::CString;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create shared memory
    let shmem = ShmemConf::new()
        .size(1024)
        .flink(CString::new("my_shared_memory")?)
        .create()?;

    // Write to shared memory
    let slice = unsafe { shmem.as_slice_mut() };
    slice[0..4].copy_from_slice(b"test");

    // In another process:
    let shmem2 = ShmemConf::new()
        .flink(CString::new("my_shared_memory")?)
        .open()?;
    println!("{:?}", &shmem2.as_slice()[0..4]);

    Ok(())
}
*/

use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::io::Write;

pub fn readOutput(fileName: String) -> std::io::Result<()> {
    // Open the same shared memory file as Python
    // Linux path; Windows uses \\.\Global\my_shm
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/shm/".to_string() + &fileName)?;

    let mut mmap = unsafe { MmapMut::map_mut(&file)? };

    // Read integers
    let data: &[i32] = unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const i32, 10) };
    println!("Rust reads: {:?}", data);
    Ok(())
}
