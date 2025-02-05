mod mmio_info;
mod memory_controller;
mod interrupt_controller;

use mmio_info::MMIOInfo;
use memory_controller::MemoryController;
use interrupt_controller::InterruptController;
use std::io::Result;
use std::path::PathBuf;
use std::time::{Instant, Duration};
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;

// Status register bits
const STATUS_DEVICE_READY: u32     = 1 << 0;
const STATUS_OPERATION_ACTIVE: u32 = 1 << 1;
const STATUS_ERROR: u32           = 1 << 2;
const STATUS_RESET_IN_PROGRESS: u32 = 1 << 3;

// Control register bits
const CONTROL_RESET: u32          = 1 << 0;

struct DeviceState {
    start_time: Instant,
    last_doorbell: u32,
    status: u32,
}

impl DeviceState {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            last_doorbell: 0,
            status: STATUS_DEVICE_READY,  // Start in ready state
        }
    }

    fn get_cycles(&self) -> u64 {
        // Convert elapsed time to cycles (assuming 1MHz clock for simplicity)
        self.start_time.elapsed().as_micros() as u64
    }

    fn update_status(&mut self, status: u32) {
        self.status = status;
    }
}

fn perform_reset(mem: &mut MemoryController, state: &mut DeviceState) -> Result<()> {
    println!("[DEV] Performing device reset");
    
    // Set reset in progress status
    state.update_status(STATUS_RESET_IN_PROGRESS);
    mem.write_register("status", state.status)?;

    // Reset all registers to default values
    mem.write_register("doorbell", 0)?;
    mem.write_register("cmd_head", 0)?;
    mem.write_register("cmd_tail", 0)?;
    mem.write_register("resp_head", 0)?;
    
    // Reset cycle counter
    state.start_time = Instant::now();
    let cycles = state.get_cycles();
    mem.write_register("cycles_low", cycles as u32)?;
    mem.write_register("cycles_high", (cycles >> 32) as u32)?;

    // Reset control register
    mem.write_register("control", 0)?;

    // Small delay to simulate hardware reset time
    std::thread::sleep(Duration::from_millis(10));

    // Set device ready status
    state.update_status(STATUS_DEVICE_READY);
    mem.write_register("status", state.status)?;

    println!("[DEV] Reset complete");
    Ok(())
}

fn main() -> Result<()> {
    let config_path = PathBuf::from(std::env::var("MMIO_INFO_PATH").unwrap_or("mmio_info.json".to_string()));
    let mmio_info = MMIOInfo::load(&config_path.to_string_lossy())?;

    let mut mem = MemoryController::new("/tmp/dev_kiwi", mmio_info.clone())?;
    let interrupt_controller = InterruptController::new("/tmp/kiwi_client.pid");
    let mut state = DeviceState::new();

    // Create PID file for client to find us
    let pid = std::process::id();
    std::fs::write("/tmp/kiwi_device.pid", pid.to_string())?;

    let (name, version) = mmio_info.get_device_info();
    println!("[DEV] {} v{} Ready (PID: {})", name, version, pid);

    // Initialize registers
    mem.write_register("status", state.status)?;  // Set device ready
    let cycles = state.get_cycles();
    mem.write_register("cycles_low", cycles as u32)?;
    mem.write_register("cycles_high", (cycles >> 32) as u32)?;

    let (cmd_base, cmd_size) = mmio_info.get_region("command_ring")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "command_ring not found"))?;
    let (resp_base, resp_size) = mmio_info.get_region("response_ring")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "response_ring not found"))?;

    loop {
        // Check if reset is requested
        if let Ok(control) = mem.read_register("control") {
            if control & CONTROL_RESET != 0 {
                perform_reset(&mut mem, &mut state)?;
                continue;
            }
        }

        // Update cycle counter periodically
        let cycles = state.get_cycles();
        mem.write_register("cycles_low", cycles as u32)?;
        mem.write_register("cycles_high", (cycles >> 32) as u32)?;

        // Check if doorbell has been rung
        let current_doorbell = mem.check_doorbell()?;
        if current_doorbell != state.last_doorbell {
            // Set operation active status
            state.update_status(state.status | STATUS_OPERATION_ACTIVE);
            mem.write_register("status", state.status)?;

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
                state.last_doorbell = current_doorbell;

                // Sync memory before raising interrupt
                mem.sync()?;

                // Send SIGUSR1 to client if PID file exists
                match std::fs::read_to_string("/tmp/kiwi_client.pid") {
                    Ok(pid_str) => {
                        if let Ok(client_pid) = pid_str.trim().parse::<i32>() {
                            match signal::kill(Pid::from_raw(client_pid), Signal::SIGUSR1) {
                                Ok(_) => println!("[DEV] Successfully sent interrupt to client"),
                                Err(e) => println!("[DEV] Note: Failed to send interrupt: {}", e)
                            }
                        }
                    },
                    Err(_) => {
                        println!("[DEV] Note: Client PID file not found - client likely disconnected");
                        // Don't set error status as this is a normal occurrence during shutdown
                    }
                }

                // Always clear operation active status and set ready, regardless of interrupt result
                state.update_status(STATUS_DEVICE_READY);
                mem.write_register("status", state.status)?;

            } else {
                // Set error status if message too large
                state.update_status(state.status | STATUS_ERROR);
                mem.write_register("status", state.status)?;
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
