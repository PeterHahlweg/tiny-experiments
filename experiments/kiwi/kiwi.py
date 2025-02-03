import mmap
import json
import time
import struct
import os
import signal
from typing import Optional, Dict
import threading

class KiwiClient:
    def __init__(self, config_path: str = "mmio_info.json", device_path: str = "/tmp/dev_kiwi", debug: bool = False):
        self.device_path = device_path
        self.sync_id = 1  # We are 1, device is 0
        self.debug = debug
       
        # Read device PID from file
        try:
            with open("/tmp/kiwi_device.pid", "r") as f:
                self.device_pid = int(f.read().strip())
            if self.debug:
                print(f"Found device PID: {self.device_pid}")
        except FileNotFoundError:
            raise RuntimeError("Device PID file not found. Is the kiwi device running?")
        except ValueError:
            raise RuntimeError("Invalid device PID format")

        # Set up response event and signal handler
        self.response_ready = threading.Event()
        self.setup_signal_handler()
        
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

        # Write our PID to shared memory so device can signal us
        our_pid = os.getpid()
        self.write_register('client_pid', our_pid)
        if self.debug:
            print(f"Registered client PID: {our_pid}")

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'fd'):
            os.close(self.fd)

    def setup_signal_handler(self):
        """Set up signal handler for simulated interrupts"""
        def signal_handler(signum, frame):
            if signum == signal.SIGUSR1:
                if self.debug:
                    print("Received interrupt signal")
                self.response_ready.set()
                
        signal.signal(signal.SIGUSR1, signal_handler)

    def _read_u32(self, offset: int) -> int:
        """Read a 32-bit value from memory"""
        self.mmap.seek(offset)
        return struct.unpack('<I', self.mmap.read(4))[0]

    def _write_u32(self, offset: int, value: int):
        """Write a 32-bit value to memory"""
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
        other_process = 1 - self.sync_id
        
        # Set our flag
        self.write_register('lock_flags', 1, self.sync_id)
        # Give priority to other process
        self.write_register('lock_turn', other_process)
        
        # Wait while other process is in critical section and has priority
        while (self.read_register('lock_flags', other_process) == 1 and 
               self.read_register('lock_turn') == other_process):
            time.sleep(0.001)

    def release_lock(self):
        """Release the lock"""
        self.write_register('lock_flags', 0, self.sync_id)

    def send_message(self, message: str, timeout_sec: float = 1.0) -> Optional[str]:
        """Send a message to the device and wait for interrupt"""
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
            
            # Clear response ready flag
            self.response_ready.clear()
            
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
                print("Waiting for interrupt...")
            
            # Wait for interrupt (signal)
            if self.response_ready.wait(timeout_sec):
                if self.debug:
                    print("Got signal, reading response...")
                    
                # Read response head value
                resp_head = self.read_register('resp_head')
                if resp_head > 0:
                    # Calculate actual response position using modulo
                    resp_pos = resp_head % resp_region['size']
                    resp_start = resp_region['base'] + resp_pos
                    
                    if self.debug:
                        print(f"Reading response at offset {resp_pos} (base + {resp_start})")
                    
                    self.mmap.seek(resp_start)
                    resp_len = struct.unpack('<I', self.mmap.read(4))[0]
                    
                    if resp_len > resp_region['size'] - 4:
                        if self.debug:
                            print(f"Invalid response length: {resp_len}")
                        return None
                        
                    resp_data = self.mmap.read(resp_len)
                    
                    # Reset response head
                    self.write_register('resp_head', 0)
                    
                    try:
                        response = resp_data.decode('utf-8')
                        if self.debug:
                            print(f"Got response: {response!r}")
                        return response
                    except UnicodeDecodeError:
                        if self.debug:
                            print("Invalid UTF-8 in response")
                        return None
                else:
                    if self.debug:
                        print("Response head is 0")
                    return None
            else:
                if self.debug:
                    print("Response timeout")
                return None
                
        finally:
            self.release_lock()

def main():
    # Example usage
    client = KiwiClient(debug=True)
    
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
