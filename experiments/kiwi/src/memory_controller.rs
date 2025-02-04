use memmap2::{MmapMut, MmapOptions};
use std::fs::{OpenOptions, File};
use std::io::{Write, Result, Error, ErrorKind};
use std::sync::atomic::{AtomicU32, Ordering};
use std::ptr;
use crate::mmio_info::MMIOInfo;

pub struct MemoryController {
    _file: File,
    mmap: MmapMut,
    mmio_info: MMIOInfo,
}

impl MemoryController {
    pub fn new(path: &str, mmio_info: MMIOInfo) -> Result<Self> {
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
            mmio_info,
        })
    }

    fn get_atomic_ptr(&self, offset: usize) -> &AtomicU32 {
        unsafe {
            let ptr = self.mmap.as_ptr().add(offset) as *const AtomicU32;
            &*ptr
        }
    }

    fn get_atomic_ptr_mut(&mut self, offset: usize) -> &mut AtomicU32 {
        unsafe {
            let ptr = self.mmap.as_mut_ptr().add(offset) as *mut AtomicU32;
            &mut *ptr
        }
    }

    pub fn read_register(&self, name: &str) -> Result<u32> {
        let reg = self.mmio_info.get_register(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, 
                format!("register '{}' not found", name)))?;
        
        let atomic = self.get_atomic_ptr(reg.offset);
        Ok(atomic.load(Ordering::Acquire))
    }

    pub fn write_register(&mut self, name: &str, value: u32) -> Result<()> {
        let reg = self.mmio_info.get_register(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, 
                format!("register '{}' not found", name)))?;
        
        let atomic = self.get_atomic_ptr_mut(reg.offset);
        atomic.store(value, Ordering::Release);
        Ok(())
    }

    pub fn ring_doorbell(&mut self) -> Result<()> {
        let reg = self.mmio_info.get_register("doorbell")
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "doorbell register not found"))?;
        
        let atomic = self.get_atomic_ptr_mut(reg.offset);
        atomic.fetch_add(1, Ordering::Release);
        Ok(())
    }

    pub fn check_doorbell(&self) -> Result<u32> {
        let reg = self.mmio_info.get_register("doorbell")
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "doorbell register not found"))?;
        
        let atomic = self.get_atomic_ptr(reg.offset);
        Ok(atomic.load(Ordering::Acquire))
    }

    pub fn read_u32(&self, offset: usize) -> Result<u32> {
        if offset + 4 > self.mmio_info.get_total_size() {
            return Err(Error::new(ErrorKind::InvalidInput, "Invalid offset"));
        }
        
        Ok(unsafe { 
            ptr::read_volatile(self.mmap.as_ptr().add(offset) as *const u32)
        })
    }

    pub fn write_u32(&mut self, offset: usize, value: u32) -> Result<()> {
        if offset + 4 > self.mmio_info.get_total_size() {
            return Err(Error::new(ErrorKind::InvalidInput, "Invalid offset"));
        }
        
        unsafe {
            ptr::write_volatile(
                self.mmap.as_mut_ptr().add(offset) as *mut u32,
                value
            );
        }
        Ok(())
    }

    pub fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        if offset + len > self.mmio_info.get_total_size() {
            return Err(Error::new(ErrorKind::InvalidInput, "Invalid offset or length"));
        }
        
        let mut result = vec![0u8; len];
        unsafe {
            ptr::copy_nonoverlapping(
                self.mmap.as_ptr().add(offset),
                result.as_mut_ptr(),
                len
            );
        }
        Ok(result)
    }

    pub fn write_bytes(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > self.mmio_info.get_total_size() {
            return Err(Error::new(ErrorKind::InvalidInput, "Invalid offset or data size"));
        }
        
        unsafe {
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.mmap.as_mut_ptr().add(offset),
                data.len()
            );
        }
        Ok(())
    }

    pub fn sync(&mut self) -> Result<()> {
        std::sync::atomic::fence(Ordering::SeqCst);
        self.mmap.flush()
    }
}
