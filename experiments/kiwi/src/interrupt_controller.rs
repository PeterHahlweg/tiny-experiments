use std::fs;
use std::io::Result;
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;

pub struct InterruptController {
    client_pid_path: String,
}

impl InterruptController {
    pub fn new(client_pid_path: &str) -> Self {
        Self {
            client_pid_path: client_pid_path.to_string(),
        }
    }
    
    /// Raise an interrupt to notify the client
    pub fn raise_irq(&self, irq: u32) -> Result<()> {
        // Map IRQ numbers to signals (currently only one IRQ supported)
        if irq != 1 {
            return Ok(());  // Ignore unknown IRQs
        }

        // Read client PID
        match fs::read_to_string(&self.client_pid_path) {
            Ok(pid_str) => {
                if let Ok(client_pid) = pid_str.trim().parse::<i32>() {
                    match signal::kill(Pid::from_raw(client_pid), Signal::SIGUSR1) {
                        Ok(_) => Ok(()),
                        Err(e) => Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to send interrupt: {}", e)
                        ))
                    }
                } else {
                    Ok(())  // Invalid PID, silently ignore
                }
            },
            Err(e) => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read client PID: {}", e)
            ))
        }
    }
}
