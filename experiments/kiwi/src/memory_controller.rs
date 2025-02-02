use memmap2::{MmapMut, MmapOptions};
use std::fs::{OpenOptions, File};
use std::io::{Write, Result, Error, ErrorKind};
use crate::mmio_info::{MMIOInfo, Register};

pub struct MemoryController {
    _file: File,
    mmap: MmapMut,
    process_id: u32,
    mmio_info: MMIOInfo,
}

impl MemoryController {
    pub fn new(path: &str, mmio_info: MMIOInfo, process_id: u32) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
            
        file.set_len(mmio_info.get_total_size() as u64)?;
        file.write_all(&vec![0; mmio_info.get_total_size()])?;
        file.sync_all()?;

        let mmap = unsafe {
            MmapOptions::new()
                .len(mmio_info.get_total_size())
                .map_mut(&file)?
        };

        Ok(Self { 
            _file: file, 
            mmap,
            process_id,
            mmio_info,
        })
    }

    fn validate_register_access(&self, name: &str, reg: &Register, index: usize, access_size: usize) -> Result<()> {
        if reg.size != access_size {
            return Err(Error::new(ErrorKind::InvalidInput, 
                format!("Register '{}': Invalid access size - tried to access {} bytes from a {}-byte register", 
                    name, access_size, reg.size)));
        }
        if index >= reg.array_size {
            return Err(Error::new(ErrorKind::InvalidInput,
                format!("Register '{}': Invalid array index {} - register array size is {}", 
                    name, index, reg.array_size)));
        }
        Ok(())
    }

    pub fn read_register_array(&self, name: &str, index: usize) -> Result<u32> {
        let reg = self.mmio_info.get_register(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, 
                format!("register '{}' not found", name)))?;
        
        self.validate_register_access(name, &reg, index, 4)?;
        let offset = reg.get_element_offset(index)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "invalid array index"))?;
            
        self.read_u32(offset)
    }

    pub fn write_register_array(&mut self, name: &str, index: usize, value: u32) -> Result<()> {
        let reg = self.mmio_info.get_register(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, 
                format!("register '{}' not found", name)))?;
        
        self.validate_register_access(name, &reg, index, 4)?;
        let offset = reg.get_element_offset(index)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "invalid array index"))?;
            
        self.write_u32(offset, value)
    }

    pub fn read_register(&self, name: &str) -> Result<u32> {
        self.read_register_array(name, 0)
    }

    pub fn write_register(&mut self, name: &str, value: u32) -> Result<()> {
        self.write_register_array(name, 0, value)
    }

    pub fn acquire_lock(&mut self) -> Result<()> {
        let other = 1 - self.process_id;
        
        // Write to our lock flag in the array
        self.write_register_array("lock_flags", self.process_id as usize, 1)?;
        self.write_register("lock_turn", other)?;
        
        // Check other process's flag and turn
        while self.read_register_array("lock_flags", other as usize)? == 1 
            && self.read_register("lock_turn")? == other {
            std::thread::sleep(std::time::Duration::from_micros(1));
        }

        Ok(())
    }

    pub fn release_lock(&mut self) -> Result<()> {
        self.write_register_array("lock_flags", self.process_id as usize, 0)
    }

    pub fn read_u32(&self, offset: usize) -> Result<u32> {
        if offset + 4 > self.mmio_info.get_total_size() {
            return Err(Error::new(ErrorKind::InvalidInput, "Invalid offset"));
        }
        
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.mmap[offset..offset + 4]);
        Ok(u32::from_le_bytes(buf))
    }

    pub fn write_u32(&mut self, offset: usize, value: u32) -> Result<()> {
        if offset + 4 > self.mmio_info.get_total_size() {
            return Err(Error::new(ErrorKind::InvalidInput, "Invalid offset"));
        }
        
        self.mmap[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    pub fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        if offset + len > self.mmio_info.get_total_size() {
            return Err(Error::new(ErrorKind::InvalidInput, "Invalid offset or length"));
        }
        
        Ok(self.mmap[offset..offset + len].to_vec())
    }

    pub fn write_bytes(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > self.mmio_info.get_total_size() {
            return Err(Error::new(ErrorKind::InvalidInput, "Invalid offset or data size"));
        }
        
        self.mmap[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    pub fn sync(&mut self) -> Result<()> {
        self.mmap.flush()
    }

    pub fn get_region(&self, name: &str) -> Option<(usize, usize)> {
        self.mmio_info.get_region(name)
    }

    pub fn get_device_info(&self) -> (&str, &str) {
        self.mmio_info.get_device_info()
    }
}
