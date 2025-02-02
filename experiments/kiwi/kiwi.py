import mmap
import json
import time
import struct
import os
from typing import Optional, Tuple, Dict

class KiwiClient:
    def __init__(self, config_path: str = "mmio_info.json", device_path: str = "/tmp/dev_kiwi", debug: bool = False):
        self.device_path = device_path
        self.process_id = 1  # We are process 1, device is process 0
        self.debug = debug
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.total_size = config['total_size']
        self.regions = {}
        self.registers = {}
        
        # Parse regions and registers
        for name, region in config['regions'].items():
            if name == 'registers':
                for reg_name, reg in region.get('registers', {}).items():
                    array_size = 2 if reg_name == 'lock_flags' else reg.get('array_size', 1)
                    self.registers[reg_name] = {
                        'offset': reg['offset'],
                        'array_size': array_size
                    }
            else:
                self.regions[name] = {
                    'base': region['base_address'],
                    'size': region['size']
                }

        # Open and map the device file
        if not os.path.exists(device_path):
            raise FileNotFoundError(f"Device file {device_path} not found. Is the kiwi accelerator running?")
            
        self.fd = os.open(device_path, os.O_RDWR)
        self.mmap = mmap.mmap(self.fd, self.total_size, mmap.MAP_SHARED)

    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'fd'):
            os.close(self.fd)

    def _read_u32(self, offset: int) -> int:
        self.mmap.seek(offset)
        return struct.unpack('<I', self.mmap.read(4))[0]

    def _write_u32(self, offset: int, value: int):
        self.mmap.seek(offset)
        self.mmap.write(struct.pack('<I', value))
        
    def read_register(self, name: str, index: int = 0) -> int:
        """Read a register value"""
        if name not in self.registers:
            raise KeyError(f"Register '{name}' not found")
        
        reg = self.registers[name]
        if index >= reg['array_size']:
            raise IndexError(f"Register '{name}': Invalid array index {index} - register array size is {reg['array_size']}")
            
        offset = reg['offset'] + (index * 4)
        value = self._read_u32(offset)
        
        if self.debug and name not in ['lock_flags', 'lock_turn', 'resp_head']:  # Don't log polling
            print(f"Read {name}[{index}] = {value}")
        return value
        
    def write_register(self, name: str, value: int, index: int = 0):
        """Write a value to a register"""
        if name not in self.registers:
            raise KeyError(f"Register '{name}' not found")
            
        reg = self.registers[name]
        if index >= reg['array_size']:
            raise IndexError(f"Register '{name}': Invalid array index {index} - register array size is {reg['array_size']}")
            
        offset = reg['offset'] + (index * 4)
        self._write_u32(offset, value)
        
        if self.debug and name not in ['lock_flags', 'lock_turn']:  # Don't log locking
            print(f"Write {name}[{index}] = {value}")

    def acquire_lock(self):
        """Acquire lock using Peterson's algorithm"""
        other_process = 1 - self.process_id
        
        # Set our flag
        self.write_register('lock_flags', 1, self.process_id)
        # Give priority to other process
        self.write_register('lock_turn', other_process)
        
        # Wait while other process is in critical section and has priority
        while (self.read_register('lock_flags', other_process) == 1 and 
               self.read_register('lock_turn') == other_process):
            time.sleep(0.001)

    def release_lock(self):
        """Release the lock"""
        self.write_register('lock_flags', 0, self.process_id)

    def send_message(self, message: str, timeout_sec: float = 1.0) -> Optional[str]:
        """Send a message to the device and get its response"""
        msg_bytes = message.encode('utf-8')
        msg_len = len(msg_bytes)
        
        cmd_region = self.regions['command_ring']
        resp_region = self.regions['response_ring']
        
        if msg_len + 4 > cmd_region['size']:
            raise ValueError(f"Message too long ({msg_len} bytes)")
            
        try:
            if self.debug:
                print(f"\nSending message: {message!r} ({msg_len} bytes)")
                
            self.acquire_lock()
            
            # Check if there's space in command ring
            cmd_head = self.read_register('cmd_head')
            cmd_tail = self.read_register('cmd_tail')
            
            if cmd_head != cmd_tail:
                if self.debug:
                    print("Ring buffer busy (head != tail)")
                return None
                
            # Write message length and data
            cmd_start = cmd_region['base'] + cmd_tail
            self.mmap.seek(cmd_start)
            self.mmap.write(struct.pack('<I', msg_len))
            self.mmap.write(msg_bytes)
            
            # Update tail pointer
            new_tail = (cmd_tail + msg_len + 4) % cmd_region['size']
            self.write_register('cmd_tail', new_tail)
            
            if self.debug:
                print("Waiting for response...")
                
            # Wait for and read response
            start_time = time.time()
            while time.time() - start_time < timeout_sec:
                resp_head = self.read_register('resp_head')
                if resp_head != 0:  # We have a response
                    resp_start = resp_region['base']
                    self.mmap.seek(resp_start)
                    resp_len = struct.unpack('<I', self.mmap.read(4))[0]
                    resp_data = self.mmap.read(resp_len)
                    
                    # Reset response head
                    self.write_register('resp_head', 0)
                    
                    response = resp_data.decode('utf-8')
                    if self.debug:
                        print(f"Got response: {response!r}")
                    return response
                    
                time.sleep(0.01)
                
            if self.debug:
                print("Response timeout")
            return None  # Timeout
            
        finally:
            self.release_lock()

def main():
    # Example usage
    client = KiwiClient(debug=True)
    
    # Print register info
    print("\nRegister configuration:")
    for name, reg in client.registers.items():
        print(f"  {name}: offset={reg['offset']}, array_size={reg['array_size']}")
    
    # Send some test messages
    test_messages = [
        "Hello, Kiwi!",
        "Testing 1 2 3",
        "How are you?"
    ]
    
    for msg in test_messages:
        response = client.send_message(msg)
        if response:
            print(f"Response: {response}")
        else:
            print("No response (timeout or busy)")
        time.sleep(0.1)  # Small delay between messages

if __name__ == "__main__":
    main()
