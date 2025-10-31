use ash::khr::workgroup_memory_explicit_layout;

use super::proc_ram;
use crate::database;
use crate::database::*;
use crate::gpu::*;

use core::panic;
use std::mem;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
pub struct Controller {}

impl Controller {
    pub fn new() -> Self {
        Controller {}
    }
    pub fn run(&self) {
        let database = database::sqlite_database::SqliteDatabase::new(String::from("mydb"));
        let table_names: Vec<String> = database
            .connection()
            .prepare("SELECT name FROM sqlite_master WHERE type='table'")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();

        println!("ALL TABLES IN DB: {:?}", table_names);

        let mut inputData = Vec::<Vec<f64>>::new();
        let mut inputData1 = Vec::<f64>::new();
        let mut inputData2 = Vec::<f64>::new();
        let mut inputData3 = Vec::<f64>::new();
        let mut stmt = database
            .connection()
            .prepare("SELECT * FROM \"temp_table\"")
            .unwrap();
        let mut rows = stmt.query([]).unwrap();
        while let Some(row) = rows.next().unwrap() {
            let t1: f64 = row.get(0).unwrap();
            let t2: f64 = row.get(1).unwrap();
            let t3: f64 = row.get(2).unwrap();
            let t4: f64 = row.get(3).unwrap();
            let t5: f64 = row.get(4).unwrap();
            inputData1.push(t1);
            inputData2.push(t2);
            inputData3.push(t3);
            //inputData.push(vec![t1, t2, t3, t4, t5]);
            //println!("{} {} {} {} {}", t1, t2, t3, t4, t5)
        }
        inputData.push(inputData1);
        inputData.push(inputData2);
        inputData.push(inputData3);
        let mut stmt = database
            .connection()
            .prepare("SELECT count(*) FROM \"temp_table\"")
            .unwrap();
        let mut rows: rusqlite::Rows<'_> = stmt.query([]).unwrap();
        let mut vv: u64 = 0;
        while let Some(row) = rows.next().unwrap() {
            let t1: u64 = row.get(0).unwrap();
            vv = t1;
        }
        let savedState = saved_state::SavedState::new();
        let shared_state = Arc::new(savedState);

        let sizeRequirements = (mem::size_of::<f64>() * 5 * vv as usize);
        println!("total size: {}", sizeRequirements);
        let b = shared_state.fetch_vram();
        println!("total x compute: {}", b.0[0]);
        println!("total invocations: {}", b.1);
        println!("total vram: {}", b.2);
        println!("total no_ram: {}", b.3);
        println!("found_local_fastest_mem: {}", b.4);
        println!("total found_local_fastest_mem: {}", b.5);

        if !b.4 {
            panic!("too old of a gpu");
        }

        proc_ram::fetchRam();
        let (loop_iter, workgroupCount, workgroupThreads) = Self::build_workgroups(
            10000000 as f64,
            3 as f64,
            b.0[0] as f64,
            b.1 as f64,
            b.5 as f64,
        );
        let mut handles: Vec<JoinHandle<()>> = vec![];
        // 8 is the queue count for the queue family do it later
        let mut free_queues: Vec<u32> = (0..8).collect();
        let (sender, receiver): (mpsc::Sender<u32>, mpsc::Receiver<u32>) = mpsc::channel();
        loop {
            let mut ff = 0;
            while ff < free_queues.len() {
                println!("{} :{}", free_queues[ff], ff);
                let q = free_queues[ff];
                let inputData_1 = inputData.clone();
                let workgroupThreads_1 = workgroupThreads;
                let workgroupCount_1 = workgroupCount;
                let v2: shaders_exec::ShaderExec = shaders_exec::ShaderExec::new(
                    String::from("x0a"),
                    Arc::clone(&shared_state),
                    workgroupThreads as u32,
                );
                let sender_ = sender.clone();
                handles.push(thread::spawn(move || {
                    v2.exec_shader(inputData_1, workgroupCount_1, workgroupThreads_1 as u32, q);
                    sender_.send(q);
                }));
                free_queues.remove(ff);
            }
            match receiver.try_recv() {
                Ok(freed) => {
                    println!("Notification received: {}", freed);
                    free_queues.push(freed);
                }
                Err(mpsc::TryRecvError::Empty) => {
                    //println!("Still working...");
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    println!("Thread ended without sending notification");
                    break;
                }
            }
        }
        /*
        for cc in 0..100 {
            // caution 8 is the total queue count for compute family
            let mut handles: Vec<JoinHandle<()>> = vec![];
            for cc2 in 0..1 {
                let inputData_1 = inputData.clone();
                let workgroupThreads_1 = workgroupThreads;
                let workgroupCount_1 = workgroupCount;
                let v2: shaders_exec::ShaderExec = shaders_exec::ShaderExec::new(
                    String::from("x0a"),
                    Arc::clone(&shared_state),
                    workgroupThreads as u32,
                );
                handles.push(thread::spawn(move || {
                    v2.exec_shader(
                        inputData_1,
                        workgroupCount_1,
                        workgroupThreads_1 as u32,
                        cc2 as u32,
                    );
                }));
            }
            for ii in handles {
                ii.join();
            }
        } */
        /*
        for iterations in 0..100 {
            //loop_iter
            let inputData_1 = inputData.clone();
            let inputData_2 = inputData.clone();
            let workgroupThreads_1 = workgroupThreads;
            let workgroupThreads_2 = workgroupThreads;
            let workgroupCount_1 = workgroupCount;
            let workgroupCount_2 = workgroupCount;
            let v2: shaders_exec::ShaderExec = shaders_exec::ShaderExec::new(
                String::from("x0a"),
                Arc::clone(&shared_state),
                workgroupThreads as u32,
            );
            let v3: shaders_exec::ShaderExec = shaders_exec::ShaderExec::new(
                String::from("x0a"),
                Arc::clone(&shared_state),
                workgroupThreads as u32,
            );
            let handle = thread::spawn(move || {
                v2.exec_shader(
                    inputData_1,
                    workgroupCount_1,
                    workgroupThreads_1 as u32,
                    0u32,
                );
            });
            let handle2 = thread::spawn(move || {
                v3.exec_shader(
                    inputData_2,
                    workgroupCount_2,
                    workgroupThreads_2 as u32,
                    1u32,
                );
            });
            handle.join();
            handle2.join();
        } */
        // v2.exec_shader(inputData, workgroupCount, workgroupThreads as u32);
    }

    fn build_workgroups(
        mut total_rows: f64,
        total_columns: f64,
        max_computeX: f64,
        mut max_invocations: f64,
        total_vram: f64,
    ) -> (u64, u64, u64) {
        const rtx2050_cores: u32 = 2048;
        // set temp assumption
        let mut row_size = total_rows;
        let mut total_iter = 1f64;
        println!("total_iter {}", total_iter);
        println!("row_size {}", row_size);
        // reduce to maxComputeX-1
        if row_size >= max_computeX {
            row_size = max_computeX - 1.0;
            total_iter = (total_rows / row_size).ceil();
        }
        println!("total_iter {}", total_iter);
        println!("row_size {}", row_size);
        // reduce to memory  limit
        // plus 1 for output buffer
        // also remove 1 gb  from vram keep it empty for gpu purpose
        let totalSizeInclOutput =
            (mem::size_of::<f64>() * total_rows as usize * (total_columns + 1.0) as usize) as f64;
        let mut total_iter_mem = (totalSizeInclOutput / (total_vram - 1024.0)).ceil();
        total_iter = (total_iter + total_iter_mem) - 1.0;
        row_size = (total_rows / total_iter).ceil();
        println!("total_iter {}", total_iter);
        println!("row_size {}", row_size);

        // now split the row size to workgroup_count and its threads
        let mut spread1 = (row_size / rtx2050_cores as f64).ceil();
        println!("spread1 {}", spread1);
        // reduce load on gpu
        max_invocations /= 2.0;
        let mut inc = 0u32;
        let mut closest = 0u64;
        let mut gap = spread1;
        let mut per_workgroup_threads = 1;
        for inc in 0..max_invocations as u32 {
            let powered = 2u64.pow(inc);
            if powered > max_invocations as u64 {
                break;
            }
            let new_gap = (spread1 as i64 - powered as i64).abs() as f64;
            if new_gap < gap {
                let mut inc2: i32 = inc as i32 - 1;
                if inc2 < 0 {
                    inc2 = 0;
                }
                per_workgroup_threads = 2u64.pow(inc2 as u32);
                gap = new_gap;
            } else {
                break;
            }
            println!("power_2 {} {}", powered, gap);
        }
        row_size = (row_size / per_workgroup_threads as f64).ceil();
        total_iter = (total_rows / (row_size * per_workgroup_threads as f64)).ceil();
        println!("total_iter {}", total_iter);
        println!("row_size {}", row_size);
        println!("per_workgroup_threads {}", per_workgroup_threads);
        (
            total_iter as u64,
            row_size as u64,
            per_workgroup_threads as u64,
        )
    }
}
