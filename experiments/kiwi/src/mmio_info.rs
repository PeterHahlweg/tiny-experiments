use std::collections::HashMap;
use std::io::{Result, Error, ErrorKind};

#[derive(Debug)]
pub struct MMIOInfo {
    device_name: String,
    version: String,
    total_size: usize,
    registers: HashMap<String, Register>,
    regions: HashMap<String, Region>,
}

#[derive(Debug, Clone)]
pub struct Register {
    pub(crate) offset: usize,
    pub(crate) size: usize,
    pub(crate) array_size: usize,  // 1 for single registers, >1 for array registers
}

impl Register {
    pub fn get_element_offset(&self, index: usize) -> Option<usize> {
        if index < self.array_size {
            Some(self.offset + (index * self.size))
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Region {
    base: usize,
    size: usize,
}

#[derive(serde::Deserialize)]
struct JsonMMIOInfo {
    device_name: String,
    version: String,
    total_size: usize,
    regions: HashMap<String, JsonRegion>,
}

#[derive(serde::Deserialize)]
struct JsonRegion {
    base_address: usize,
    size: usize,
    #[serde(default)]
    registers: Option<HashMap<String, JsonRegister>>,
}

#[derive(serde::Deserialize)]
struct JsonRegister {
    offset: usize,
    #[serde(default = "default_array_size")]
    array_size: usize,
}

fn default_array_size() -> usize {
    1  // Default to non-array register, except for special cases handled below
}

impl MMIOInfo {
    pub fn load(path: &str) -> Result<Self> {
        let config_file = std::fs::read_to_string(path)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to read MMIO info file: {}", e)))?;
            
        let config: JsonMMIOInfo = serde_json::from_str(&config_file)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to parse MMIO info: {}", e)))?;

        let mut registers = HashMap::new();
        if let Some(reg_region) = config.regions.get("registers") {
            if let Some(regs) = &reg_region.registers {
                for (name, reg) in regs {
                    let array_size = if name == "lock_flags" {
                        2  // lock_flags needs size 2 for two processes
                    } else {
                        reg.array_size
                    };
                    
                    registers.insert(name.clone(), Register {
                        offset: reg.offset,
                        size: 4,  // All registers are 4 bytes (u32)
                        array_size,
                    });
                }
            }
        }

        let mut regions = HashMap::new();
        for (name, region) in config.regions {
            if name != "registers" {
                regions.insert(name, Region {
                    base: region.base_address,
                    size: region.size,
                });
            }
        }

        Ok(Self {
            device_name: config.device_name,
            version: config.version,
            total_size: config.total_size,
            registers,
            regions,
        })
    }

    pub fn get_register(&self, name: &str) -> Option<Register> {
        self.registers.get(name).cloned()
    }

    pub fn get_region(&self, name: &str) -> Option<(usize, usize)> {
        self.regions.get(name).map(|r| (r.base, r.size))
    }

    pub fn get_total_size(&self) -> usize {
        self.total_size
    }

    pub fn get_device_info(&self) -> (&str, &str) {
        (&self.device_name, &self.version)
    }
}
