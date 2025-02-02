mod mmio_info;
mod memory_controller;

use mmio_info::MMIOInfo;
use memory_controller::MemoryController;
use std::io::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let config_path = PathBuf::from(std::env::var("MMIO_INFO_PATH").unwrap_or("mmio_info.json".to_string()));
    let mmio_info = MMIOInfo::load(&config_path.to_string_lossy())?;
    
    let mut mem = MemoryController::new("/tmp/dev_kiwi", mmio_info, 0)?;

    let (name, version) = mem.get_device_info();
    println!("[DEV] {} v{} Ready", name, version);
    
    let (cmd_base, cmd_size) = mem.get_region("command_ring")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "command_ring not found"))?;
    let (resp_base, _) = mem.get_region("response_ring")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "response_ring not found"))?;

    loop {
        mem.acquire_lock()?;
        
        let cmd_head = mem.read_register("cmd_head")?;
        let cmd_tail = mem.read_register("cmd_tail")?;

        if cmd_head != cmd_tail {
            // Read message length and data from command ring
            let msg_len = mem.read_u32(cmd_base + cmd_head as usize)?;
            let msg = mem.read_bytes(cmd_base + cmd_head as usize + 4, msg_len as usize)?;
            
            // Print received message if it's valid UTF-8
            if let Ok(msg_str) = String::from_utf8(msg.clone()) {
                println!("[DEV] Got: {}", msg_str);
            }

            // Write to response ring
            let resp_head = mem.read_register("resp_head")?;
            let resp_start = resp_base + resp_head as usize;
            
            mem.write_u32(resp_start, msg_len)?;
            mem.write_bytes(resp_start + 4, &msg)?;

            // Update ring buffer pointers
            mem.write_register("resp_head", (resp_head + 4 + msg_len) % cmd_size as u32)?;
            mem.write_register("cmd_head", (cmd_head + 4 + msg_len) % cmd_size as u32)?;
            
            mem.sync()?;
        }
        
        mem.release_lock()?;
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
