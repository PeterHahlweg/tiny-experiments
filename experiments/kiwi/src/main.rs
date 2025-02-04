mod mmio_info;
mod memory_controller;

use mmio_info::MMIOInfo;
use memory_controller::MemoryController;
use std::io::Result;
use std::path::PathBuf;
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;

fn main() -> Result<()> {
    let config_path = PathBuf::from(std::env::var("MMIO_INFO_PATH").unwrap_or("mmio_info.json".to_string()));
    let mmio_info = MMIOInfo::load(&config_path.to_string_lossy())?;

    let mut mem = MemoryController::new("/tmp/dev_kiwi", mmio_info.clone())?;

    // Create PID file for client to find us
    let pid = std::process::id();
    std::fs::write("/tmp/kiwi_device.pid", pid.to_string())?;

    let (name, version) = mmio_info.get_device_info();
    println!("[DEV] {} v{} Ready (PID: {})", name, version, pid);

    let (cmd_base, cmd_size) = mmio_info.get_region("command_ring")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "command_ring not found"))?;
    let (resp_base, resp_size) = mmio_info.get_region("response_ring")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "response_ring not found"))?;

    let mut last_doorbell = mem.check_doorbell()?;

    loop {
        // Check if doorbell has been rung
        let current_doorbell = mem.check_doorbell()?;
        if current_doorbell != last_doorbell {
            // Read message length and data from command ring
            let msg_len = mem.read_u32(cmd_base)?;
            let msg = mem.read_bytes(cmd_base + 4, msg_len as usize)?;

            // Print received message if it's valid UTF-8
            if let Ok(msg_str) = String::from_utf8(msg.clone()) {
                println!("[DEV] Got: {}", msg_str);
            }

            // Write to response ring
            let msg_total_len = (msg_len + 4) as usize;
            if msg_total_len <= resp_size {
                // Write response
                mem.write_u32(resp_base, msg_len)?;
                mem.write_bytes(resp_base + 4, &msg)?;

                // Update last seen doorbell value
                last_doorbell = current_doorbell;

                // Sync memory
                mem.sync()?;

                // Send SIGUSR1 to client
                if let Ok(client_pid) = mem.read_register("client_pid") {
                    if client_pid != 0 {
                        println!("[DEV] Sending SIGUSR1 to client PID: {}", client_pid);
                        match signal::kill(Pid::from_raw(client_pid as i32), Signal::SIGUSR1) {
                            Ok(_) => println!("[DEV] Successfully sent signal to client"),
                            Err(e) => println!("[DEV] Failed to send signal: {}", e)
                        }
                    }
                }
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
