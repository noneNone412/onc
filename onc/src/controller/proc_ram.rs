use std::fs;

pub fn fetchRam() -> std::io::Result<()> {
    let meminfo = fs::read_to_string("/proc/meminfo")?;

    let mut mem_total_kb = 0;
    let mut mem_available_kb = 0;

    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            mem_total_kb = line
                .split_whitespace()
                .nth(1)
                .unwrap()
                .parse::<u64>()
                .unwrap();
        } else if line.starts_with("MemAvailable:") {
            mem_available_kb = line
                .split_whitespace()
                .nth(1)
                .unwrap()
                .parse::<u64>()
                .unwrap();
        }
    }

    let mem_total_mb = mem_total_kb / 1024;
    let mem_available_mb = mem_available_kb / 1024;

    println!("Total RAM: {} MB", mem_total_mb);
    println!("Available RAM: {} MB", mem_available_mb);
    println!("Used RAM: {} MB", mem_total_mb - mem_available_mb);

    Ok(())
}
