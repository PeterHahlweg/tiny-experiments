from typing import Optional, Dict
import mmap
import json
import time
import struct
import os
import signal
import ctypes

class AtomicU32(ctypes.Structure):
    _fields_ = [("value", ctypes.c_uint32)]

class KiwiClient:
    def __init__(self, config_path: str = "mmio_info.json", device_path: str = "/tmp/dev_kiwi", debug: bool = False):
        self.device_path = device_path
        self.debug = debug
        self.response_ready = False
        self._atomic_refs = []  # Keep track of atomic references

        # Read device PID from file
        try:
            with open("/tmp/kiwi_device.pid", "r") as f:
                self.device_pid = int(f.read().strip())
            if self.debug:
                print(f"Device PID: {self.device_pid}")
        except FileNotFoundError:
            raise RuntimeError("Device PID file not found. Is the kiwi device running?")
        except ValueError:
            raise RuntimeError("Invalid device PID format")

        # Set up signal handler for interrupts
        signal.signal(signal.SIGUSR1, self._handle_response)

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
                    self.registers[reg_name] = {
                        'offset': reg['offset'],
                        'array_size': reg.get('array_size', 1)
                    }
            else:
                self.regions[name] = {
                    'base': region['base_address'],
                    'size': region['size']
                }

        # Open and map the device file
        if not os.path.exists(device_path):
            raise FileNotFoundError(f"Device file {device_path} not found")

        self.fd = os.open(device_path, os.O_RDWR)
        self.mmap = mmap.mmap(self.fd, self.total_size, mmap.MAP_SHARED)

        # Map atomic registers
        self.atomic_regs = {}
        for name, reg in self.registers.items():
            offset = reg['offset']
            atomic = AtomicU32.from_buffer(self.mmap, offset)
            self.atomic_regs[name] = atomic
            self._atomic_refs.append(atomic)  # Keep reference

        # Write our PID to shared memory
        self.write_register('client_pid', os.getpid())

        if self.debug:
            print("Initialization complete")

    def __del__(self):
        """Cleanup resources"""
        # Clear atomic references first
        self.atomic_regs.clear()
        self._atomic_refs.clear()
        
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'fd'):
            os.close(self.fd)

    def _handle_response(self, signum, frame):
        """Signal handler for SIGUSR1"""
        if signum == signal.SIGUSR1:
            self.response_ready = True

    def read_register(self, name: str) -> int:
        """Read a register value atomically"""
        if name not in self.atomic_regs:
            raise KeyError(f"Register '{name}' not found")
        return self.atomic_regs[name].value

    def write_register(self, name: str, value: int):
        """Write a value to a register atomically"""
        if name not in self.atomic_regs:
            raise KeyError(f"Register '{name}' not found")
        self.atomic_regs[name].value = value

    def ring_doorbell(self):
        """Signal the device that new data is available"""
        current = self.atomic_regs['doorbell'].value
        self.atomic_regs['doorbell'].value = current + 1

    def send_message(self, message: str, timeout_sec: float = 1.0) -> Optional[str]:
        """Send a message to the device and wait for response"""
        msg_bytes = message.encode('utf-8')
        msg_len = len(msg_bytes)

        cmd_region = self.regions['command_ring']
        resp_region = self.regions['response_ring']

        if msg_len + 4 > cmd_region['size']:
            raise ValueError(f"Message too long ({msg_len} bytes)")

        try:
            if self.debug:
                print(f"Request: {message}")

            # Block signals while we set up the request and wait for response
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
            try:
                # Reset response flag
                self.response_ready = False
                
                # Write message to command ring
                cmd_start = cmd_region['base']
                self.mmap.seek(cmd_start)
                self.mmap.write(struct.pack('<I', msg_len))
                self.mmap.write(msg_bytes)

                # Ensure writes are visible
                self.mmap.flush()

                # Signal device
                self.ring_doorbell()
                
                # Temporarily unblock signals to receive the response
                signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR1})
                
                # Wait for interrupt signal
                start_time = time.time()
                while time.time() - start_time < timeout_sec:
                    if self.response_ready:
                        # Block signals while reading response
                        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
                        
                        # Read and return response (only once)
                        self.mmap.seek(resp_region['base'])
                        resp_len = struct.unpack('<I', self.mmap.read(4))[0]

                        if resp_len > 0 and resp_len <= resp_region['size'] - 4:
                            resp_data = self.mmap.read(resp_len)
                            try:
                                response = resp_data.decode('utf-8')
                                if self.debug:
                                    print(f"Response: {response}")
                                return response
                            except UnicodeDecodeError as e:
                                if self.debug:
                                    print(f"Unicode decode error: {e}")
                                return None
                        
                        # No need to unblock signals here since we're returning
                        return None
                            
                    time.sleep(0.001)  # Short sleep to prevent busy waiting

                if self.debug:
                    print(f"Timeout after {timeout_sec}s")
                return None

            finally:
                # Restore original signal mask
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

        except Exception as e:
            if self.debug:
                print(f"Error in send_message: {e}")
            raise

def main():
    """Example usage"""
    client = KiwiClient(debug=True)

    test_messages = [
        "Hello, Kiwi!",
        "Testing 1 2 3",
        "How are you?"
    ]

    for msg in test_messages:
        print(f"\nSending: {msg}")
        response = client.send_message(msg)
        if not response: print("No response (timeout)")
        time.sleep(0.1)

if __name__ == "__main__":
    main()
