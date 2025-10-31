// runs shaders on the gpu
use super::saved_state::SavedState;
use std::fs;

use std::fs::File;
use std::io::Read;

use ash::khr::workgroup_memory_explicit_layout;
use ash::vk::{Handle, SpecializationInfo, StructureType};
use serde_json::Value;

use ash::*;
use ash::{ext::debug_utils, nv::glsl_shader};

use std::{
    ffi::{CStr, CString},
    mem::{align_of, offset_of, size_of, size_of_val},
    os::raw::{c_char, c_void},
    path::Path,
    ptr::copy_nonoverlapping,
};

use std::ptr;

use std::collections::HashSet;
use std::sync::Arc;

use std::mem;

#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    x_count: u32,
}

pub struct ShaderExec {
    path: String,
    ss: Arc<SavedState>,
    shaderModule: ash::vk::ShaderModule,
    inputDesciptorSetLayout: ash::vk::DescriptorSetLayout,
    outputDescriptorSetLayout: ash::vk::DescriptorSetLayout,
    descriptorPool: ash::vk::DescriptorPool,
    inputVarCount: u32,
    outputVarCount: u32,
    commandPool: ash::vk::CommandPool,
    commandBufferExecEqn: Vec<ash::vk::CommandBuffer>,
    pushConstantRangeForInputLength: ash::vk::PushConstantRange,
    computePipelineLayout: ash::vk::PipelineLayout,
    computePipeline: Vec<ash::vk::Pipeline>,
    fence: ash::vk::Fence,
}

impl ShaderExec {
    pub fn new(name: String, savedState: Arc<SavedState>, workgroupThreads: u32) -> Self {
        // on create make permanent variables:
        let file_paths = Self::openFiles(name.as_str());
        let shaderModule = Self::create_shaderModule(savedState.device(), file_paths.0.as_str());
        let (
            inputDesciptorSetLayout,
            outputDescriptorSetLayout,
            descriptorPool,
            inputVarCount,
            outputVarCount,
        ) = Self::create_descriptorSetLayoutAndPool(savedState.device(), file_paths.1.as_str());
        let commandPool =
            Self::create_command_pool(*savedState.computeQueue(), savedState.device());
        let commandBufferExecEqn =
            Self::create_command_buffer_exec_eqn(&commandPool, savedState.device());
        const totalRows: u32 = 1;
        let pushConstantRangeForInputLength =
            Self::create_push_constant_for_input_length(totalRows);

        let (computePipelineLayout, computePipeline) = Self::create_compute_pipline(
            savedState.device(),
            pushConstantRangeForInputLength,
            shaderModule,
            workgroupThreads,
            &inputDesciptorSetLayout,
            &outputDescriptorSetLayout,
        );
        let fence = Self::create_fence(savedState.device());
        println!("Regular expressions");
        ShaderExec {
            path: name,
            ss: savedState,
            shaderModule: shaderModule,
            inputDesciptorSetLayout: inputDesciptorSetLayout,
            outputDescriptorSetLayout: outputDescriptorSetLayout,
            descriptorPool: descriptorPool,
            inputVarCount: inputVarCount,
            outputVarCount: outputVarCount,
            commandPool: commandPool,
            commandBufferExecEqn: commandBufferExecEqn,
            pushConstantRangeForInputLength: pushConstantRangeForInputLength,
            computePipelineLayout: computePipelineLayout,
            computePipeline: computePipeline,
            fence: fence,
        }
    }

    // pub fn write_buffers(&self) -> (vk::Buffer, vk::Buffer) {}
    pub fn exec_shader(
        &self,
        data: Vec<Vec<f64>>,
        workgroups: u64,
        workgroup_threads: u32,
        queueIndex: u32,
    ) {
        // Sum the lengths of all inner Vec<f32> to get total elements
        let total_elements: usize = data.iter().map(|v| v.len()).sum();
        // Calculate required buffer size in bytes
        let individual_buffer_size = (std::mem::size_of::<f64>() * data[0].len()) as vk::DeviceSize;
        println!(
            "llllllllllllllllll----- {} {}",
            data[0].len(),
            individual_buffer_size
        );
        // count of outputs
        let result_count = data[0].len();
        // reset command buffers
        let result_reset = unsafe {
            self.ss.device().reset_command_buffer(
                self.commandBufferExecEqn[0],
                vk::CommandBufferResetFlags::empty(),
            )
        };
        if result_reset.is_err() {
            panic!("paniced at exec_shader");
        }
        // reset  descriptor pool
        unsafe {
            self.ss
                .device()
                .reset_descriptor_pool(self.descriptorPool, vk::DescriptorPoolResetFlags::empty())
                .expect("Failed to reset descriptor pool");
        }
        let layouts = [self.inputDesciptorSetLayout, self.outputDescriptorSetLayout];
        // allocate  descriptor sets
        let alloc_info: vk::DescriptorSetAllocateInfo = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: self.descriptorPool,
            descriptor_set_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let descriptor_sets: Vec<vk::DescriptorSet> = unsafe {
            self.ss
                .device()
                .allocate_descriptor_sets(&alloc_info)
                .expect("Failed to allocate descriptor sets")
        };
        // update the above allocated descriptorSet with buffers
        // create and allocate input buffers
        let mut input_buffers: Vec<vk::Buffer> = Vec::new();
        let mut input_buffers_memory: Vec<vk::DeviceMemory> = Vec::new();
        let mut input_descriptor_writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        let mut input_descriptor_buffer_info: Vec<vk::DescriptorBufferInfo> = Vec::new();

        // vector.len() = rows
        for i in 0..self.inputVarCount as u32 {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(individual_buffer_size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = unsafe { self.ss.device().create_buffer(&buffer_info, None) };
            if buffer.is_err() {
                panic!("inputBuffer_{} creation failed", i);
            }
            let mem_requirements = unsafe {
                self.ss
                    .device()
                    .get_buffer_memory_requirements(buffer.unwrap())
            };
            let mem_properties = unsafe {
                self.ss
                    .instance()
                    .get_physical_device_memory_properties(*self.ss.physicalDevice())
            };

            let mem_type_index = self.find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                mem_properties,
            );

            // Allocate memory
            let allocate_info: vk::MemoryAllocateInfo = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: mem_type_index,
                _marker: std::marker::PhantomData,
            };
            let buffer_memory = unsafe {
                self.ss
                    .device()
                    .allocate_memory(&allocate_info, None)
                    .expect("Failed to allocate buffer memory")
            };
            // Bind memory
            unsafe {
                self.ss
                    .device()
                    .bind_buffer_memory(buffer.unwrap(), buffer_memory, 0)
                    .expect("Failed to bind buffer memory");
            }
            // Map memory and copy data
            unsafe {
                let data_ptr = unsafe {
                    self.ss
                        .device()
                        .map_memory(
                            buffer_memory,
                            0,
                            individual_buffer_size as u64,
                            vk::MemoryMapFlags::empty(),
                        )
                        .expect("Failed to map memory") as *mut f64
                };
                // Copy the i-th input vector to GPU memory
                //println!("this  is  the buffer data {:?}", input_data[i as usize]);
                std::ptr::copy_nonoverlapping(
                    data[i as usize].as_ptr(),
                    data_ptr,
                    data[i as usize].len(),
                );

                self.ss.device().unmap_memory(buffer_memory);
                // Store buffer and memory handles
                input_buffers.push(buffer.unwrap());
                input_buffers_memory.push(buffer_memory); // need to maintain this vector

                let descriptorBufferInfo = vk::DescriptorBufferInfo {
                    buffer: buffer.unwrap(),
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                };
                input_descriptor_buffer_info.push(descriptorBufferInfo);
                // write descriptor sets
                let writeDescriptorSet: vk::WriteDescriptorSet = vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_sets[0], //input set
                    dst_binding: i as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_image_info: std::ptr::null(),
                    p_buffer_info: &input_descriptor_buffer_info[i as usize],
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                };
                input_descriptor_writes.push(writeDescriptorSet);
            }
        }

        // create ouput buffer that acts as a container for result
        let mut output_buffers: Vec<vk::Buffer> = Vec::new();
        let mut output_buffers_memory: Vec<vk::DeviceMemory> = Vec::new();
        let mut output_descriptor_writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        // kept it 0 for initial write
        let output_data_temp: Vec<f64> = vec![0.0; data[0].len()];
        for i in 0..self.outputVarCount as u32 {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(individual_buffer_size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = unsafe { self.ss.device().create_buffer(&buffer_info, None) };
            if buffer.is_err() {
                panic!("outputBuffer_{} creation failed", i);
            }
            let mem_requirements = unsafe {
                self.ss
                    .device()
                    .get_buffer_memory_requirements(buffer.unwrap())
            };
            let mem_properties = unsafe {
                self.ss
                    .instance()
                    .get_physical_device_memory_properties(*self.ss.physicalDevice())
            };

            let mem_type_index = self.find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                mem_properties,
            );

            // Allocate memory
            let allocate_info: vk::MemoryAllocateInfo = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: mem_type_index,
                _marker: std::marker::PhantomData,
            };
            let buffer_memory = unsafe {
                self.ss
                    .device()
                    .allocate_memory(&allocate_info, None)
                    .expect("Failed to allocate buffer memory")
            };
            // Bind memory
            unsafe {
                self.ss
                    .device()
                    .bind_buffer_memory(buffer.unwrap(), buffer_memory, 0)
                    .expect("Failed to bind buffer memory");
            }
            // Map memory and copy data
            unsafe {
                let data_ptr = self
                    .ss
                    .device()
                    .map_memory(
                        buffer_memory,
                        0,
                        individual_buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to map memory") as *mut f64;

                // Copy the i-th output vector to GPU memory
                std::ptr::copy_nonoverlapping(
                    output_data_temp.as_ptr(),
                    data_ptr,
                    data[i as usize].len(),
                );
                self.ss.device().unmap_memory(buffer_memory);
                // Store buffer and memory handles
                output_buffers.push(buffer.unwrap());
                output_buffers_memory.push(buffer_memory); // You'll need to maintain this vector

                // bind descriptor set  to buffers
                let descriptorBufferInfo: vk::DescriptorBufferInfo = vk::DescriptorBufferInfo {
                    buffer: buffer.unwrap(),
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                };
                // write descriptor sets
                let writeDescriptorSet: vk::WriteDescriptorSet = vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descriptor_sets[1], //output set
                    dst_binding: i as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_image_info: std::ptr::null(),
                    p_buffer_info: &descriptorBufferInfo,
                    p_texel_buffer_view: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                };
                output_descriptor_writes.push(writeDescriptorSet);
            }
        }
        // update input descriptor sets
        unsafe {
            self.ss
                .device()
                .update_descriptor_sets(&input_descriptor_writes, &[]);
        };
        // update output descriptor sets
        unsafe {
            self.ss
                .device()
                .update_descriptor_sets(&output_descriptor_writes, &[]);
        };
        // begin command buffer
        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        unsafe {
            self.ss
                .device()
                .begin_command_buffer(self.commandBufferExecEqn[0], &command_buffer_begin_info)
                .expect("Failed to begin command buffer");
        }
        // bind pipeline
        unsafe {
            self.ss.device().cmd_bind_pipeline(
                self.commandBufferExecEqn[0],
                vk::PipelineBindPoint::COMPUTE,
                self.computePipeline[0],
            );
        }
        // bind descriptor set
        let descriptor_sets2 = [
            descriptor_sets[0], // set = 0: input buffer set
            descriptor_sets[1], // set = 1: output buffer set
        ];
        unsafe {
            self.ss.device().cmd_bind_descriptor_sets(
                self.commandBufferExecEqn[0],
                vk::PipelineBindPoint::COMPUTE,
                self.computePipelineLayout,
                0u32,
                &descriptor_sets2,
                &[],
            );
        }
        // bind push constants = total x_params
        let push_constants = PushConstants {
            x_count: result_count as u32,
        };
        unsafe {
            self.ss.device().cmd_push_constants(
                self.commandBufferExecEqn[0],
                self.computePipelineLayout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    (&push_constants as *const PushConstants) as *const u8,
                    std::mem::size_of::<PushConstants>(),
                ),
            );
        }

        // cmd dispatch
        unsafe {
            self.ss
                .device()
                .cmd_dispatch(self.commandBufferExecEqn[0], workgroups as u32, 1, 1);
        }
        // end recording command buffers
        unsafe {
            self.ss
                .device()
                .end_command_buffer(self.commandBufferExecEqn[0]);
        }
        // reset fence
        unsafe {
            self.ss.device().reset_fences(&[self.fence]);
        }
        // submit task on gpu

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0u32,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1u32,
            p_command_buffers: self.commandBufferExecEqn.as_ptr(),
            signal_semaphore_count: 0u32,
            p_signal_semaphores: ptr::null(),
            _marker: std::marker::PhantomData,
        };
        unsafe {
            self.ss
                .device()
                .reset_fences(&[self.fence])
                .expect("Failed to reset fence");
        }

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: self.commandBufferExecEqn.as_ptr(),
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let compute_queue = unsafe {
            self.ss
                .device()
                .get_device_queue(*self.ss.computeQueue(), queueIndex)
        };
        unsafe {
            self.ss
                .device()
                .queue_submit(compute_queue, &[submit_info], self.fence)
                .expect("Failed to submit to queue");
        }
        println!("one");
        let status = unsafe { self.ss.device().get_fence_status(self.fence) };
        println!("Fence status: {:?}", status);
        unsafe {
            self.ss
                .device()
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .expect("Failed to wait for fence");
        }
        println!("one");
        let status = unsafe { self.ss.device().get_fence_status(self.fence) };
        println!("Fence status: {:?}", status);
        // 1. Copy from GPU output buffer to staging buffer
        let cmd_buf = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.commandPool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            unsafe { self.ss.device().allocate_command_buffers(&alloc_info) }.unwrap()[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let (staginBuffer, stagginfDeviceMemory, stagingPtr) =
            self.create_staging_resources(individual_buffer_size);
        unsafe {
            self.ss
                .device()
                .begin_command_buffer(cmd_buf, &begin_info)
                .unwrap();

            let copy_region = vk::BufferCopy::default().size(individual_buffer_size);
            self.ss.device().cmd_copy_buffer(
                cmd_buf,
                output_buffers[0], // GPU buffer
                staginBuffer,      // HOST_CACHED staging buffer
                &[copy_region],
            );

            self.ss.device().end_command_buffer(cmd_buf).unwrap();
        }

        // 2. Submit with fence
        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buf));

        unsafe {
            self.ss.device().reset_fences(&[self.fence]).unwrap();
            self.ss
                .device()
                .queue_submit(
                    compute_queue, // use transfer queue
                    &[submit_info],
                    self.fence,
                )
                .unwrap();

            self.ss
                .device()
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .unwrap();
        }

        // 3. Read directly from persistently mapped pointer
        let mut output_data = vec![0f64; result_count as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(stagingPtr, output_data.as_mut_ptr(), output_data.len());
        }
        // now destroy all
        unsafe {
            self.ss.device().destroy_buffer(staginBuffer, None);
            self.ss.device().free_memory(stagginfDeviceMemory, None);
        }
        for (buffer, memory) in input_buffers.iter().zip(input_buffers_memory.iter()) {
            unsafe {
                self.ss.device().destroy_buffer(*buffer, None);
                self.ss.device().free_memory(*memory, None);
            }
        }
        for (buffer, memory) in output_buffers.iter().zip(output_buffers_memory.iter()) {
            unsafe {
                self.ss.device().destroy_buffer(*buffer, None);
                self.ss.device().free_memory(*memory, None);
            }
        }
        unsafe {
            self.ss
                .device()
                .free_command_buffers(self.commandPool, &[cmd_buf]);
        }

        // Print results
        /* for val in &output_data {
            print!("{} ", val);
        } */
        /*
        unsafe {
            // 2. Map memory
            let data_ptr = self
                .ss
                .device()
                .map_memory(
                    output_buffers_memory[0],
                    0,
                    individual_buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *const f64;

            // Read into a Vec
            let mut output_data: Vec<f64> = vec![0f64; result_count as usize];
            std::ptr::copy_nonoverlapping(data_ptr, output_data.as_mut_ptr(), output_data.len());
            // Unmap memory
            self.ss.device().unmap_memory(output_buffers_memory[0]);

            for ff in 0..output_data.len() {
                let gg = output_data[ff];
                print!("{} ", gg);
            }
            // println!("output_data {}", output_data.len());
        };
        */
        println!("exec_shader");
    }
    fn create_staging_resources(
        &self,
        size: vk::DeviceSize,
    ) -> (ash::vk::Buffer, ash::vk::DeviceMemory, *const f64) {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.ss.device().create_buffer(&buffer_info, None) }
            .expect("Failed to create staging buffer");

        let mem_reqs = unsafe { self.ss.device().get_buffer_memory_requirements(buffer) };

        let mem_props = unsafe {
            self.ss
                .instance()
                .get_physical_device_memory_properties(*self.ss.physicalDevice())
        };

        // More flexible memory type selection
        let mem_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                let flags = mem_props.memory_types[i as usize].property_flags;
                flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
                    && (mem_reqs.memory_type_bits & (1 << i)) != 0
            })
            .expect("No suitable memory type found") as u32;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let buffer_memory = unsafe { self.ss.device().allocate_memory(&alloc_info, None) }
            .expect("Failed to allocate staging memory");

        unsafe {
            self.ss
                .device()
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind staging memory");
        }

        let ptr = unsafe {
            self.ss
                .device()
                .map_memory(buffer_memory, 0, size, vk::MemoryMapFlags::empty())
                .expect("Failed to map staging memory")
        } as *const f64;

        (buffer, buffer_memory, ptr)
    }
    fn copy_gpu_to_cpu(
        &self,
        device: &ash::Device,
        cmd_pool: vk::CommandPool,
        queue: vk::Queue,
        src_buffer: vk::Buffer,     // device-local GPU buffer
        staging_buffer: vk::Buffer, // HOST_CACHED staging buffer
        copy_size: vk::DeviceSize,
        fence: vk::Fence,
    ) {
        // Allocate a short-lived command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buf = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(cmd_buf, &begin_info).unwrap();

            // Copy
            let copy_region = vk::BufferCopy::default().size(copy_size);
            device.cmd_copy_buffer(cmd_buf, src_buffer, staging_buffer, &[copy_region]);

            device.end_command_buffer(cmd_buf).unwrap();
        }

        // Submit with fence
        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buf));

        unsafe {
            device.reset_fences(&[fence]).unwrap();
            device.queue_submit(queue, &[submit_info], fence).unwrap();
        }
    }
}
impl ShaderExec {
    fn openFiles(path: &str) -> (String, String) {
        let cargo_toml_dir = env!("CARGO_MANIFEST_DIR");
        let compiledShaderPath = format!(
            "{}/src/gpu/shaders/compiled/{}{}",
            cargo_toml_dir, path, ".spv"
        );
        let jsonShaderPath = format!(
            "{}/src/gpu/shaders/json/{}{}",
            cargo_toml_dir, path, ".json"
        );
        (compiledShaderPath, jsonShaderPath)
    }
}
impl ShaderExec {
    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> u32 {
        for (i, mem_type) in mem_properties.memory_types
            [..mem_properties.memory_type_count as usize]
            .iter()
            .enumerate()
        {
            if (type_filter & (1 << i)) != 0 && mem_type.property_flags.contains(properties) {
                return i as u32;
            }
        }
        panic!("Failed to find suitable memory type!");
    }
}

impl ShaderExec {
    fn value_to_u32(value: &Value) -> Option<u32> {
        match value {
            Value::Number(n) => n.as_u64().and_then(|num| {
                if num <= u32::MAX as u64 {
                    Some(num as u32)
                } else {
                    None
                }
            }),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }
    fn create_shaderModule(device: &ash::Device, compiledPath: &str) -> ash::vk::ShaderModule {
        println!("{}", compiledPath);
        let mut file = File::open(compiledPath);
        match &file {
            Ok(f) => {
                println!("file opened");
            }
            Err(e) => {
                eprintln!("failed to create shader module");
            }
        }
        let mut buffer = Vec::new();
        file.unwrap().read_to_end(&mut buffer);
        // 32 bytes
        let spv_data = unsafe {
            // Check alignment and length
            if buffer.as_ptr() as usize % mem::align_of::<u32>() != 0
                || buffer.len() % mem::size_of::<u32>() != 0
            {}

            // SAFETY: We've checked alignment and size
            std::slice::from_raw_parts(
                buffer.as_ptr() as *const u32,
                buffer.len() / mem::size_of::<u32>(),
            )
        };
        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(spv_data);
        let shaderModule: Result<vk::ShaderModule, vk::Result> =
            unsafe { device.create_shader_module(&shader_module_info, None) };
        match &shaderModule {
            Ok(s) => *s,
            Err(e) => {
                panic!("failed to create shader module");
            }
        }
    }
    fn create_descriptorSetLayoutAndPool(
        device: &ash::Device,
        jsonPath: &str,
    ) -> (
        ash::vk::DescriptorSetLayout,
        ash::vk::DescriptorSetLayout,
        ash::vk::DescriptorPool,
        u32,
        u32,
    ) {
        let mut inputBindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::new();
        let mut outputBindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::new();
        let mut inOutBindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::new();
        let mut input_vars_count: u32 = 0;
        let mut output_vars_count: u32 = 0;
        let json_content = fs::read_to_string(jsonPath).expect("Failed to read file");
        let json_data: Value = serde_json::from_str(&json_content).expect("Failed to parse JSON");
        if let Some(array_a) = json_data.get("inputDescriptor").and_then(|v| v.as_array()) {
            for item in array_a {
                let binding_num = Self::value_to_u32(&item["binding"]).unwrap();
                let bindingIn = vk::DescriptorSetLayoutBinding {
                    binding: binding_num, // Binding index in the shader
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER, // Type of descriptor (e.g., uniform buffer)
                    descriptor_count: 1, // Number of descriptors for this binding
                    stage_flags: vk::ShaderStageFlags::COMPUTE, // Shader stages that use this binding
                    p_immutable_samplers: std::ptr::null(),     // Optional samplers, if required
                    _marker: std::marker::PhantomData, // This ensures the proper lifetime for the struct
                };
                inputBindings.push(bindingIn);
                inOutBindings.push(bindingIn);
                input_vars_count += 1;
            }
        }

        let nonine = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(), // Or your specific flags
            binding_count: inputBindings.len() as u32,
            p_bindings: inputBindings.as_ptr(),
            _marker: std::marker::PhantomData,
        };
        let input_descriptor_set_layouts: Result<vk::DescriptorSetLayout, vk::Result> =
            unsafe { device.create_descriptor_set_layout(&nonine, None) };
        match &input_descriptor_set_layouts {
            Ok(t) => {}
            Err(e) => {
                panic!("input descriptor set layouts not found");
            }
        }
        if let Some(array_a) = json_data.get("outputDescriptor").and_then(|v| v.as_array()) {
            for item in array_a {
                let binding_num = Self::value_to_u32(&item["binding"]).unwrap();
                let bindingOut = vk::DescriptorSetLayoutBinding {
                    binding: binding_num, // Binding index in the shader
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER, // Type of descriptor (e.g., uniform buffer)
                    descriptor_count: 1, // Number of descriptors for this binding
                    stage_flags: vk::ShaderStageFlags::COMPUTE, // Shader stages that use this binding
                    p_immutable_samplers: std::ptr::null(),     // Optional samplers, if required
                    _marker: std::marker::PhantomData, // This ensures the proper lifetime for the struct
                };
                outputBindings.push(bindingOut);
                inOutBindings.push(bindingOut);
                output_vars_count += 1;
            }
        }
        let nonine = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(), // Or your specific flags
            binding_count: outputBindings.len() as u32,
            p_bindings: outputBindings.as_ptr(),
            _marker: std::marker::PhantomData,
        };
        let output_descriptor_set_layouts: Result<vk::DescriptorSetLayout, vk::Result> =
            unsafe { device.create_descriptor_set_layout(&nonine, None) };
        match &output_descriptor_set_layouts {
            Ok(t) => {}
            Err(e) => {
                panic!("output descriptor set layouts not found");
            }
        }
        let total_storage_buffers = inputBindings.len() + outputBindings.len();
        let pool_sizes = [
            // For set=0 bindings 0-2 (3 readonly storage buffers)
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: total_storage_buffers as u32,
            },
        ];
        let descriptor_pool_info: vk::DescriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            max_sets: 2u32, // one for  set=0 and other  for set =1
            pool_size_count: 1u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            _marker: std::marker::PhantomData,
        };
        let descriptor_pool: Result<vk::DescriptorPool, vk::Result> =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) };
        match &descriptor_pool {
            Ok(d) => {}
            Err(e) => {
                panic!("descriptor pool not created");
            }
        }
        (
            input_descriptor_set_layouts.unwrap(),
            output_descriptor_set_layouts.unwrap(),
            descriptor_pool.unwrap(),
            input_vars_count,
            output_vars_count,
        )
    }
    fn create_command_pool(computeQueue: u32, device: &ash::Device) -> ash::vk::CommandPool {
        let command_pool_info: vk::CommandPoolCreateInfo = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: computeQueue,
            _marker: std::marker::PhantomData,
        };
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) };
        match &command_pool {
            Ok(d) => *d,
            Err(e) => {
                panic!("command pool not created");
            }
        }
    }

    fn create_command_buffer_exec_eqn(
        command_pool: &ash::vk::CommandPool,
        device: &ash::Device,
    ) -> Vec<ash::vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: *command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            _marker: std::marker::PhantomData,
        };

        let command_buffer =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) };

        match &command_buffer {
            Ok(d) => {}
            Err(e) => {
                panic!("Command buffer not created");
            }
        }
        command_buffer.unwrap()
    }
    fn create_push_constant_for_input_length(totalRows: u32) -> ash::vk::PushConstantRange {
        (ash::vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE, // Or other stages (VERTEX, FRAGMENT, etc.)
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        })
    }

    fn create_compute_pipline(
        device: &ash::Device,
        push_constant_range: ash::vk::PushConstantRange,
        shaderModule: ash::vk::ShaderModule,
        workGroupThreads: u32,
        inputDescriptorSetLayout: &vk::DescriptorSetLayout,
        outputDescriptorSetLayout: &vk::DescriptorSetLayout,
    ) -> (ash::vk::PipelineLayout, Vec<vk::Pipeline>) {
        let specialization_constants = [vk::SpecializationMapEntry {
            constant_id: 0, // Matches the constant_id in the shader
            offset: 0,
            size: std::mem::size_of::<u32>(),
        }];

        let specialization_data = workGroupThreads.to_le_bytes(); // Value for WORKGROUP_SIZE_X

        let specialization_info = vk::SpecializationInfo {
            map_entry_count: specialization_constants.len() as u32,
            p_map_entries: specialization_constants.as_ptr(),
            data_size: specialization_data.len(),
            p_data: specialization_data.as_ptr() as *const std::ffi::c_void,
            _marker: std::marker::PhantomData,
        };
        let descriptorSetLayouts = [*inputDescriptorSetLayout, *outputDescriptorSetLayout];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: descriptorSetLayouts.len() as u32,
            p_set_layouts: descriptorSetLayouts.as_ptr(),
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            _marker: std::marker::PhantomData,
        };

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) };
        match &pipeline_layout {
            Ok(n) => {}
            Err(e) => {
                panic!("pipeine layout failed");
            }
        }
        let stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shaderModule,
            p_name: b"main\0".as_ptr() as *const i8,
            p_specialization_info: &specialization_info,
            _marker: std::marker::PhantomData,
        };
        let pipeline_create_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage: stage_create_info,
            layout: pipeline_layout.unwrap(),
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
            _marker: std::marker::PhantomData,
        };
        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
            )
        };
        match &pipeline {
            Ok(k) => {}
            Err(e) => {
                panic!("Pipeline creation Failed");
            }
        }
        (pipeline_layout.unwrap(), pipeline.unwrap())
    }

    fn create_fence(device: &ash::Device) -> ash::vk::Fence {
        let fence_info: vk::FenceCreateInfo = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
            _marker: std::marker::PhantomData,
        };

        let fence: Result<ash::vk::Fence, ash::vk::Result> =
            unsafe { device.create_fence(&fence_info, None) };
        match &fence {
            Ok(i) => {}
            Err(e) => {
                eprintln!("Failed to create fence");
                panic!("Failed to create the fence program exiting");
            }
        }
        return fence.unwrap();
    }
}
