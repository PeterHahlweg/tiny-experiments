from typing import Optional, Dict, Tuple
import mmap, json, time, struct, os, signal, ctypes

class AtomicU32(ctypes.Structure):
    _fields_ = [("value", ctypes.c_uint32)]

class KiwiClient:
    CLIENT_PID_PATH = "/tmp/kiwi_client.pid"
    
    def __init__(self, config_path: str = "mmio_info.json", device_path: str = "/tmp/dev_kiwi", debug: bool = False):
        self.debug = debug
        self.response_ready = False
        
        # Read device PID
        try:
            self.device_pid = int(open("/tmp/kiwi_device.pid").read().strip())
            self.debug and print(f"Device PID: {self.device_pid}")
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError("Device PID error. Is kiwi device running?") from e

        # Write client PID to file
        with open(self.CLIENT_PID_PATH, 'w') as f:
            f.write(str(os.getpid()))

        # Load config and parse regions
        config = json.load(open(config_path))
        self.total_size = config['total_size']
        self.regions = {k: {'base': v['base_address'], 'size': v['size']} 
                       for k, v in config['regions'].items() if k != 'registers'}
        self.registers = {k: {'offset': v['offset'], 'array_size': v.get('array_size', 1)}
                         for k, v in config['regions'].get('registers', {}).get('registers', {}).items()}

        # Set up memory mapping and atomic registers
        if not os.path.exists(device_path):
            raise FileNotFoundError(f"Device file {device_path} not found")
            
        self.fd = os.open(device_path, os.O_RDWR)
        self.mmap = mmap.mmap(self.fd, self.total_size, mmap.MAP_SHARED)
        self._atomic_refs = []
        self.atomic_regs = {}
        
        for name, reg in self.registers.items():
            atomic = AtomicU32.from_buffer(self.mmap, reg['offset'])
            self.atomic_regs[name] = atomic
            self._atomic_refs.append(atomic)

        # Initialize signal handler
        signal.signal(signal.SIGUSR1, lambda s, f: setattr(self, 'response_ready', True))
        self.debug and print("Initialization complete")

    def __del__(self):
        self.atomic_regs.clear()
        self._atomic_refs.clear()
        hasattr(self, 'mmap') and self.mmap.close()
        hasattr(self, 'fd') and os.close(self.fd)
        # Clean up PID file on exit
        try:
            os.remove(self.CLIENT_PID_PATH)
        except OSError:
            pass

    def read_register(self, name: str) -> int:
        if name not in self.atomic_regs:
            raise KeyError(f"Register '{name}' not found")
        return self.atomic_regs[name].value

    def write_register(self, name: str, value: int):
        if name not in self.atomic_regs:
            raise KeyError(f"Register '{name}' not found")
        self.atomic_regs[name].value = value

    def ring_doorbell(self):
        self.atomic_regs['doorbell'].value += 1

    def _write_command(self, cmd_region: Dict, message: bytes):
        """Write a command to the command ring"""
        self.mmap.seek(cmd_region['base'])
        self.mmap.write(struct.pack('<I', len(message)))
        self.mmap.write(message)
        self.mmap.flush()

    def _read_response(self, resp_region: Dict) -> Optional[str]:
        """Read and decode a response from the response ring"""
        self.mmap.seek(resp_region['base'])
        resp_len = struct.unpack('<I', self.mmap.read(4))[0]
        
        if resp_len <= 0 or resp_len > resp_region['size'] - 4:
            return None
            
        try:
            response = self.mmap.read(resp_len).decode('utf-8')
            self.debug and print(f"Response: {response}")
            return response
        except UnicodeDecodeError:
            self.debug and print("Unicode decode error")
            return None

    def _wait_for_response(self, timeout_sec: float) -> bool:
        """Wait for response signal with timeout"""
        start = time.time()
        while time.time() - start < timeout_sec:
            if self.response_ready:
                return True
            time.sleep(0.001)
        self.debug and print(f"Timeout after {timeout_sec}s")
        return False

    def send_message(self, message: str, timeout_sec: float = 1.0) -> Optional[str]:
        """Send a message to the device and wait for response"""
        try:
            msg_bytes = message.encode('utf-8')
            cmd_region = self.regions['command_ring']
            resp_region = self.regions['response_ring']

            if len(msg_bytes) + 4 > cmd_region['size']:
                raise ValueError(f"Message too long ({len(msg_bytes)} bytes)")

            self.debug and print(f"Request: {message}")
            
            # Lock interrupts during device communication
            with self._interrupt_lock() as irq:
                self.response_ready = False
                self._write_command(cmd_region, msg_bytes)
                self.ring_doorbell()
                
                # Temporarily unblock to receive response
                irq.unblock()  # Enable interrupts to receive device response
                if not self._wait_for_response(timeout_sec):
                    return None

                # Disable interrupts while reading device memory
                irq.block()
                return self._read_response(resp_region)

        except Exception as e:
            self.debug and print(f"Error in send_message: {e}")
            raise

    class _interrupt_lock:
        """Context manager for interrupt handling emulation"""
        def __init__(self):
            self.old_mask = None
            
        def __enter__(self):
            self.old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
            return self
            
        def block(self):
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
            
        def unblock(self):
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR1})
            
        def __exit__(self, *args):
            signal.pthread_sigmask(signal.SIG_SETMASK, self.old_mask)

def main():
    client = KiwiClient(debug=True)
    messages = ["Hello, Kiwi!", "Testing 1 2 3", "How are you?"]

    for msg in messages:
        print(f"\nSending: {msg}")
        if not client.send_message(msg):
            print("No response (timeout)")
        time.sleep(0.1)

if __name__ == "__main__":
    main()
