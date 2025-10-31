// this saved_stat.rs will save the following items
// 1) vk_instance
// 2) create a debug callback
// 3) vk_device
// 4) vk fence

//  modifiable  saved items
// 1) shader module
// 1) command pool
// 1) command buffer
// 1) compute pipeline
// 1) compute pipeline layout
// 1) descriptors pools, set_layout
// 1) vkBuffer for reading
// 1) vkBuffer for writing
// 1) workgroup count management
// using a singleton pattern

use super::debug;
use ash::ext::physical_device_drm;
use ash::{ext::debug_utils, vk, Device, Entry, Instance};
use std::collections::HashSet;
use std::default;
use std::ptr::null;
use std::{
    ffi::{CStr, CString},
    mem::{align_of, offset_of, size_of, size_of_val},
    os::raw::{c_char, c_void},
    path::Path,
};

pub struct SavedState {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_messenger_callback: vk::DebugUtilsMessengerEXT,
    physicalDevice: ash::vk::PhysicalDevice,
    physicalDeviceIndex: usize,
    device: ash::Device,
    computeQueue: u32,
    transferQueue: u32,
    fence: ash::vk::Fence,
}

impl SavedState {
    pub fn new() -> Self {
        // Statically loads Vulkan library from OS
        let entry = ash::Entry::linked();
        println!("Vulkan Entry loaded successfully!");
        // Create Vulkan instance
        let instance = match SavedState::create_instance(&entry) {
            Ok(inst) => {
                println!("Vulkan instance created successfully!");
                Some(inst)
            }
            Err(e) => {
                eprintln!("Failed to create Vulkan instance: {:?}", e);
                // stop the program here
                None
            }
        };
        // Create debug messenger
        let debug_messenger_callback =
            SavedState::setup_debug_messenger(&entry, instance.as_ref().unwrap());
        match &debug_messenger_callback {
            Ok(g) => {
                println!("Vulkan debug messenger created successfully!");
            }
            Err(e) => {
                eprintln!("Failed to create Vulkan debug messenger.");
                // stop the program here
            }
        };
        // setup device
        let (physicalDeviceIndex, physicalDevice, vkDevice, computeQueue, transferQueue) =
            SavedState::setup_vkDevice(instance.as_ref().unwrap());
        // create fence
        let fence = Self::create_fence(&vkDevice);

        SavedState {
            entry: entry,
            instance: instance.unwrap(),
            debug_messenger_callback: debug_messenger_callback.unwrap(),
            physicalDevice: physicalDevice,
            physicalDeviceIndex: physicalDeviceIndex,
            device: vkDevice,
            computeQueue: computeQueue,
            transferQueue: transferQueue,
            fence: fence,
        }
    }
}
impl SavedState {
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }
    pub fn physicalDeviceIndex(&self) -> &usize {
        &self.physicalDeviceIndex
    }
    pub fn physicalDevice(&self) -> &ash::vk::PhysicalDevice {
        &self.physicalDevice
    }
    pub fn device(&self) -> &ash::Device {
        &self.device
    }
    pub fn computeQueue(&self) -> &u32 {
        &self.computeQueue
    }
    pub fn transferQueue(&self) -> &u32 {
        &self.transferQueue
    }
    pub fn fence(&self) -> &ash::vk::Fence {
        &self.fence
    }
}
// constructor functions
impl SavedState {
    fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, ash::vk::Result> {
        let app_name = CString::new("Vulkan Application").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));

        // validation  begins
        let mut enabled_layers = Vec::new();
        let mut create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let mut layernames: Box<[*const i8]> = Box::from([]);

        if Self::check_validation_layer(&entry) {
            const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];
            enabled_layers = VALIDATION_LAYERS
                .iter()
                .map(|&s| CString::new(s).unwrap())
                .collect();

            let layer_names: Vec<*const i8> =
                enabled_layers.iter().map(|layer| layer.as_ptr()).collect();
            layernames = layer_names.into_boxed_slice();
        }
        let mut debug_create_info = Self::create_debug_create_info();
        let debug_extension = CString::new("VK_EXT_debug_utils").unwrap();
        let extension_names: [*const i8; 1] = [debug_extension.as_ptr()];
        create_info = create_info
            .enabled_layer_names(&layernames)
            .push_next(&mut debug_create_info);
        create_info = create_info.enabled_extension_names(&extension_names);

        let instance: Result<Instance, vk::Result> =
            unsafe { entry.create_instance(&create_info, None) };
        match &instance {
            Ok(inst) => {
                println!("instance created");
            }
            Err(e) => {
                println!("instance not created");
            }
        }
        return instance;
    }
    fn setup_debug_messenger(
        entry: &Entry,
        instance: &Instance,
    ) -> Result<vk::DebugUtilsMessengerEXT, vk::Result> {
        let create_info = SavedState::create_debug_create_info();
        let debug_utils = debug_utils::Instance::new(entry, instance);
        let debug_utils_messenger: Result<vk::DebugUtilsMessengerEXT, vk::Result> =
            unsafe { debug_utils.create_debug_utils_messenger(&create_info, None) };
        match &debug_utils_messenger {
            Ok(suc) => {
                println!("Debug Messenger Setup Successful");
            }
            Err(e) => {
                eprintln!("Error while calling {}", e);
            }
        }
        return debug_utils_messenger;
    }
    fn setup_vkDevice(
        instance: &ash::Instance,
    ) -> (usize, ash::vk::PhysicalDevice, ash::Device, u32, u32) {
        // pickup physical device
        let physical_devices: Result<Vec<vk::PhysicalDevice>, vk::Result> =
            unsafe { instance.enumerate_physical_devices() };
        match &physical_devices {
            Ok(i) => {}
            Err(e) => {
                panic!("code  interrupted");
            }
        };
        for (index, device) in physical_devices.unwrap().iter().enumerate() {
            let properties: vk::PhysicalDeviceProperties =
                unsafe { instance.get_physical_device_properties(*device) };
            let device_name: String = unsafe {
                std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            println!("Physical Device {}: {}", &index, &device_name);
            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                println!("is discrete {}", &device_name);
                // select compute queue and transfer queue
                let queue_family_properties =
                    unsafe { instance.get_physical_device_queue_family_properties(device.clone()) };

                let mut compute_queue_family_index: Option<u32> = None;
                let mut transfer_queue_family_index: Option<u32> = None;

                // first iterate for compute queue
                let mut available_queues = 0u32;
                for (i, qf) in queue_family_properties.iter().enumerate() {
                    let flags: vk::QueueFlags = qf.queue_flags;
                    println!(
                        "Queue family {} flags: {}{}{}{}{}",
                        i,
                        if flags.contains(vk::QueueFlags::GRAPHICS) {
                            "GRAPHICS "
                        } else {
                            ""
                        },
                        if flags.contains(vk::QueueFlags::COMPUTE) {
                            "COMPUTE "
                        } else {
                            ""
                        },
                        if flags.contains(vk::QueueFlags::TRANSFER) {
                            "TRANSFER "
                        } else {
                            ""
                        },
                        if flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                            "SPARSE_BINDING "
                        } else {
                            ""
                        },
                        if flags.contains(vk::QueueFlags::PROTECTED) {
                            "PROTECTED "
                        } else {
                            ""
                        },
                    );
                    if flags.contains(vk::QueueFlags::COMPUTE)
                        && !flags.contains(vk::QueueFlags::GRAPHICS)
                        && flags.contains(vk::QueueFlags::TRANSFER)
                    {
                        compute_queue_family_index = Some(i as u32);
                        transfer_queue_family_index = Some(i as u32);
                        println!("QUEUE family count{}", qf.queue_count);
                        available_queues = qf.queue_count;
                        break;
                    }
                }
                // first iterate for compute queue
                for (i, qf) in queue_family_properties.iter().enumerate() {
                    let flags: vk::QueueFlags = qf.queue_flags;
                    println!(
                        "Queue family {} flags: {}{}{}{}{}",
                        i,
                        if flags.contains(vk::QueueFlags::GRAPHICS) {
                            "GRAPHICS "
                        } else {
                            ""
                        },
                        if flags.contains(vk::QueueFlags::COMPUTE) {
                            "COMPUTE "
                        } else {
                            ""
                        },
                        if flags.contains(vk::QueueFlags::TRANSFER) {
                            "TRANSFER "
                        } else {
                            ""
                        },
                        if flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                            "SPARSE_BINDING "
                        } else {
                            ""
                        },
                        if flags.contains(vk::QueueFlags::PROTECTED) {
                            "PROTECTED "
                        } else {
                            ""
                        },
                    );
                    /* if flags.contains(vk::QueueFlags::TRANSFER) {
                        transfer_queue_family_index = Some(i as u32);
                        break;
                    } */
                }

                // fallbacks if dedicated  compute queue not found
                if compute_queue_family_index.is_none() {
                    for (i, qf) in queue_family_properties.iter().enumerate() {
                        if qf.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                            compute_queue_family_index = Some(i as u32);
                            break;
                        }
                    }
                }

                //
                let mut unique_queue_families = HashSet::new();
                if let Some(index) = compute_queue_family_index {
                    unique_queue_families.insert(index);
                }
                if let Some(index) = transfer_queue_family_index {
                    unique_queue_families.insert(index);
                }
                // Using the actual available queue count
                // Create priorities for each queue we want to use
                let priorities = vec![1.0f32; available_queues as usize];
                let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_queue_families
                    .iter()
                    .map(|&family| {
                        vk::DeviceQueueCreateInfo {
                            queue_family_index: family,
                            queue_count: priorities.len() as u32,
                            p_queue_priorities: priorities.as_ptr(),
                            ..Default::default() // <- fills the rest with safe Vulkan defaults
                        }
                    })
                    .collect();

                // DeviceCreateInfo
                let mut device_features = vk::PhysicalDeviceFeatures::default();
                device_features.shader_float64 = vk::TRUE;

                let device_create_info = vk::DeviceCreateInfo {
                    queue_create_info_count: queue_create_infos.len() as u32,
                    p_queue_create_infos: queue_create_infos.as_ptr(),
                    p_enabled_features: &device_features,
                    ..Default::default()
                };

                let device_created = match unsafe {
                    instance.create_device(device.clone(), &device_create_info, None)
                } {
                    Ok(device) => {
                        println!("Logical device created successfully!");
                        Some(device)
                    }
                    Err(e) => {
                        eprintln!("Failed to create logical device: {:?}", e);
                        panic!("Program interrupted");
                    }
                };
                match &compute_queue_family_index {
                    Some(i) => {
                        println!("found compute queue {}", i)
                    }
                    None => {
                        println!("Failed to find compute queue")
                    }
                }
                match &transfer_queue_family_index {
                    Some(i) => {
                        println!("found transfer queue {}", i)
                    }
                    None => {
                        println!("Failed to find transfer queue")
                    }
                }
                return (
                    index,
                    *device,
                    device_created.unwrap(),
                    compute_queue_family_index.unwrap(),
                    transfer_queue_family_index.unwrap(),
                );
                // only one gpu is to be used
                // so break;
            }
        }
        panic!("no discrete gpu found");
    }
}

impl SavedState {
    fn check_validation_layer(entry: &ash::Entry) -> bool {
        const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

        let layer_properties = match unsafe { entry.enumerate_instance_layer_properties() } {
            Ok(layers) => layers,
            Err(e) => {
                eprintln!("Failed to enumerate instance layers: {:?}", e);
                return false; // Explicitly return false on error
            }
        };

        for required_layer in VALIDATION_LAYERS {
            let mut found = false;
            for layer in &layer_properties {
                let layer_name = unsafe { std::ffi::CStr::from_ptr(layer.layer_name.as_ptr()) };
                if layer_name.to_str().unwrap() == *required_layer {
                    found = true;
                    break;
                }
            }
            if !found {
                eprintln!("Validation layer not found: {}", required_layer);
                return false; // Explicitly return false if layer not found
            }
        }
        true // Explicitly return true if all layers are found
    }

    fn create_debug_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug::vulkan_debug_callback))
    }
    fn check_validation_layer_support(entry: &Entry) {
        let supported_layers = unsafe { entry.enumerate_instance_layer_properties().unwrap() };
        for required in ["VK_LAYER_KHRONOS_validation"].iter() {
            let found = supported_layers.iter().any(|layer| {
                let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                let name = name.to_str().expect("Failed to get layer name pointer");
                required == &name
            });

            if !found {
                panic!("Validation layer not supported: {}", required);
            }
        }
    }
}
impl SavedState {
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

    pub fn fetch_vram(&self) -> ([u32; 3], u32, u64, u64, bool, u64) {
        let mem_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physicalDevice)
        };

        let mut total_vram: u64 = 0;
        let mut no_ram: u64 = 0;
        for i in 0..mem_properties.memory_heap_count as usize {
            let heap = mem_properties.memory_heaps[i];
            println!("from heap: {}", i);
            // We only count heaps that are device local (i.e. VRAM)
            if (heap.flags & vk::MemoryHeapFlags::DEVICE_LOCAL) == vk::MemoryHeapFlags::DEVICE_LOCAL
            {
                total_vram += heap.size;
            } else {
                no_ram += heap.size;
            }
        }
        let mut found_local_fastest_mem = false;
        let mut heap_size_local_fastest_mem: u64 = 0;
        println!("\nMemory Types");
        for i in 0..mem_properties.memory_type_count as usize {
            let mem_type = mem_properties.memory_types[i];
            let heap_index = mem_type.heap_index;
            let heap_size = mem_properties.memory_heaps[heap_index as usize].size;
            let size_gib = heap_size as f64 / 1_073_741_824.0;
            let size_gb = heap_size as f64 / 1_000_000_000.0;

            print!("Type {} (Heap {}): ", i, heap_index);
            print!("Size: {:.2} GiB ({:.2} GB) | ", size_gib, size_gb);

            print!("Flags: ");
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            {
                print!("DEVICE_LOCAL ");
            }
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            {
                print!("HOST_VISIBLE ");
            }
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                print!("HOST_COHERENT ");
            }
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_CACHED)
            {
                print!("HOST_CACHED ");
            }
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
            {
                print!("LAZILY_ALLOCATED ");
            }
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::PROTECTED)
            {
                print!("PROTECTED ");
            }
            println!();
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                && mem_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                && mem_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                found_local_fastest_mem = true;
                heap_size_local_fastest_mem = heap_size;
            }
        }
        let features = unsafe {
            &self
                .instance()
                .get_physical_device_features(self.physicalDevice)
        };
        let props = unsafe {
            &self
                .instance()
                .get_physical_device_properties(self.physicalDevice)
        };
        let maxComputeWorkGroupCount = props.limits.max_compute_work_group_count;
        let maxComputeWorkGroupInvocations = props.limits.max_compute_work_group_invocations;
        (
            maxComputeWorkGroupCount,
            maxComputeWorkGroupInvocations,
            total_vram,
            no_ram,
            found_local_fastest_mem,
            heap_size_local_fastest_mem,
        )
    }
}
