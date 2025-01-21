from __future__ import annotations
import socket, json, struct, traceback, threading, os, argparse, time, re
import ctypes
from ctypes import c_void_p, c_size_t, pythonapi, POINTER, c_char
from functools import partial, lru_cache
from typing import List, Any, Optional, Tuple, Dict
from tinygrad import Tensor
from tinygrad.runtime.ops_metal import MetalDevice, MetalProgram, MetalBuffer, MetalAllocator, MetalCompiler, msg
from tinygrad.runtime.graph.metal import MetalGraph
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.device import Buffer, Device, Compiled
from tinygrad.helpers import DEBUG, getenv, to_mv

class CompilationError(Exception): pass
class ExecutionError(Exception): pass

class RemoteMetalConnection:
    _instances: Dict[Tuple[str, int], 'RemoteMetalConnection'] = {}
    def __new__(cls, host="localhost", port=9123):
        key = (host, port)
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, host="localhost", port=9123, timeout=30):
        if self._initialized: return
        self.host, self.port, self.timeout = host, port, timeout
        self._socket, self._initialized = None, True
        self._connect()

    def _connect(self) -> None:
        if self._socket is None:
            try:
                if DEBUG >= 3: print(f"[NET  ] Connecting to {self.host}:{self.port}")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                self._socket = sock
                if DEBUG >= 3: print(f"[NET  ] Connection established")
            except socket.error as e:
                if self._socket: self._socket.close(); self._socket = None
                raise ConnectionError(f"Failed to connect to Metal server: {e}")

    def ensure_connection(self):
        if not self._socket: self._connect()

    def send_data(self, data: bytes) -> None:
        try:
            total_size = len(data)
            if DEBUG >= 3: print(f"[NET  ] Sending {total_size} bytes")
            self._socket.sendall(struct.pack("!I", total_size))
            for i in range(0, total_size, 8192):
                self._socket.sendall(data[i:min(i + 8192, total_size)])
        except socket.error as e:
            self.close()
            raise ConnectionError(f"Error sending data to Metal server: {e}")
        except Exception as e: raise

    def receive_data(self) -> bytes:
        try:
            if not self._socket: raise ConnectionError("No active connection")
            size_data = self._socket.recv(4)
            if not size_data or len(size_data) != 4:
                raise ConnectionError(f"Incomplete size data received: {len(size_data)} bytes")
            expected_size = struct.unpack("!I", size_data)[0]
            if DEBUG >= 3: print(f"[NET] Receiving {expected_size} bytes")
            response_data = bytearray()
            bytes_received = 0
            while bytes_received < expected_size:
                remaining = expected_size - bytes_received
                chunk = self._socket.recv(min(remaining, 8192))
                if not chunk:
                    raise ConnectionError(f"Connection closed prematurely after {bytes_received}/{expected_size} bytes")
                response_data.extend(chunk)
                bytes_received += len(chunk)
                if DEBUG >= 3: print(f"[NET] Received {bytes_received} bytes")
            return bytes(response_data)
        except socket.error as e:
            self.close()
            raise ConnectionError(f"Error receiving data from Metal server: {e}")
        except Exception as e: raise

    def close(self) -> None:
        if self._socket:
            try: self._socket.close()
            except socket.error: pass
            finally:
                self._socket = None
                key = (self.host, self.port)
                if key in self._instances: del self._instances[key]

class HostMetalCompiler(MetalCompiler):
    def __init__(self):
        if DEBUG >= 3: print(f"[COMP ] Creating compiler")
        super().__init__()

    def compile(self, src: str) -> bytes:
        try:
            if DEBUG >= 3:
                print(f"[COMP ] Starting compilation of kernel source ({len(src)} bytes)")
                print(f"[COMP ] Source code: \n{src}\n")
            result = super().compile(src)
            if DEBUG >= 3 and (m := re.search(r'kernel void (\w+)', src)):
                print(f"[COMPILE] Successfully compiled kernel '{m.group(1)}' -> {len(result)} bytes")
            return result
        except Exception as e:
            print(f"ERROR in compilation: {e}")
            raise

    @lru_cache(None)
    def compile_cached(self, src: str) -> bytes: return self.compile(src)

class TinyMetalProgram:
    def __nothing__(): None

class RemoteMetalProgram:
    def __init__(self, dev: RemoteMetalDevice, name: str, lib: bytes):
        if DEBUG >= 3: print(f"[PROG ] Creating program '{name}' ({len(lib)} bytes)")
        self.name = name
        self.lib = lib
        self.device = dev
        # Create the local program for fallback
        self.prog = TinyMetalProgram(dev, name, lib)

    @staticmethod
    def _get_buffer_data(buf):
        """Extract raw data from a MetalBuffer."""
        try:
            if DEBUG >= 3: print(f"[BUFFER] Getting data from buffer: {buf}")
            # Get raw pointer to buffer contents
            contents_ptr = msg(buf.buf, "contents")
            if not contents_ptr:
                if DEBUG >= 3: print("[BUFFER] No contents pointer returned")
                return None

            # Create memoryview and get bytes
            view = to_mv(contents_ptr, buf.size)
            raw_data = bytes(view)

            if raw_data:
                if DEBUG >= 3: print(f"[BUFFER] Successfully got buffer data: {len(raw_data)} bytes")
                if len(raw_data) >= 4:  # If we have at least one float
                    float_vals = struct.unpack('f' * (len(raw_data) // 4), raw_data)
                    if DEBUG >= 3: print(f"[BUFFER] Buffer contains float values: {float_vals}")
            return raw_data.hex() if raw_data else None
        except Exception as e:
            if DEBUG >= 3: print(f"[BUFFER] Failed to get buffer data: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def _set_buffer_data(buf, hex_data):
        """Update a MetalBuffer with new data."""
        if not hex_data: return
        try:
            raw_data = bytes.fromhex(hex_data)
            if DEBUG >= 3: print(f"[BUFFER] Setting buffer data: {len(raw_data)} bytes")
            if len(raw_data) >= 4:  # If we have at least one float
                float_vals = struct.unpack('f' * (len(raw_data) // 4), raw_data)
                if DEBUG >= 3: print(f"[BUFFER] Setting float values: {float_vals}")

            # Get pointer to buffer contents
            contents_ptr = msg(buf.buf, "contents")
            if not contents_ptr:
                if DEBUG >= 3: print("[BUFFER] Error: No contents pointer returned")
                return

            # Create memoryview and update contents
            dest = to_mv(contents_ptr, len(raw_data))
            dest[:] = raw_data
            if DEBUG >= 3: print(f"[BUFFER] Successfully updated buffer")
        except Exception as e:
            if DEBUG >= 3: print(f"[BUFFER] Failed to set buffer data: {e}")
            traceback.print_exc()

    def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
        if DEBUG >= 3:
            print(f"\n[EXEC ] Running '{self.name}' with {len(bufs)} buffers")
            print(f"[EXEC] Global={global_size}, Local={local_size}, Values={vals}")

        try:
            buffer_info = []
            for i, buf in enumerate(bufs):
                if DEBUG >= 3: print(f"\n[BUFFER] Processing buffer {i}:")
                data = RemoteMetalProgram._get_buffer_data(buf)
                info = {
                    "size": buf.size,
                    "offset": buf.offset,
                    "data": data
                }
                if DEBUG >= 3: print(f"[BUFFER] Buffer {i} info: size={buf.size}, offset={buf.offset}")
                if data:
                    if DEBUG >= 3: print(f"[BUFFER] Data length: {len(data)}")
                buffer_info.append(info)

            request = {
                "type": "kernel_execute",
                "name": self.name,
                "kernel_obj": self.lib.hex(),
                "buffer_info": buffer_info,
                "global_size": global_size,
                "local_size": local_size,
                "ops": [],
                "values": list(vals),
                "wait": wait
            }

            if DEBUG >= 3: print(f"[EXEC] Sending request to runner: {json.dumps(request, indent=2)}")
            self.device.connection.send_data(json.dumps(request).encode())
            response = json.loads(self.device.connection.receive_data().decode())

            if DEBUG >= 3: print(f"[EXEC] Got response: {response}")
            if response.get("status") != "success":
                raise ExecutionError(f"Kernel execution failed: {response.get('error')}")

            # Update buffer contents with data returned from server
            for i, (buf, updated_info) in enumerate(zip(bufs, response.get("buffers", []))):
                if "data" in updated_info:
                    if DEBUG >= 3: print(f"\n[BUFFER] Processing response for buffer {i}")
                    RemoteMetalProgram._set_buffer_data(buf, updated_info["data"])

            kernel_time = response.get("kernel_time", 0.0)
            if DEBUG >= 3 and wait: print(f"[EXEC] Completed '{self.name}' in {kernel_time:.6f}s")
            return kernel_time if wait else None

        except Exception as e:
            print(f"\nERROR in remote execution: {str(e)}")
            print(traceback.format_exc())
            print("\nFalling back to local execution...")
            return self.prog(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)

class RemoteMetalDevice(MetalDevice):
    def __init__(self, device: str, host="localhost", port=9123):
        if DEBUG >= 3: print(f"[DEV  ] Initializing Metal device for {device}")
        self.connection = RemoteMetalConnection(host=host, port=port)
        self.remote_compiler = HostMetalCompiler()
        if "METAL" not in Device._devices: Device._devices.append("METAL")
        super().__init__(device)
        self.renderer = MetalRenderer()
        self.allocator = MetalAllocator(self)
        self.compiler = self.remote_compiler
        self.graph = MetalGraph
        self.func_program = partial(self._create_program)
        if DEBUG >= 3: print(f"[DEVICE] Metal device initialized")
        os.environ["METAL"] = "1"

    @staticmethod
    def available() -> bool:
        try:
            host = os.environ.get("METAL_REMOTE_HOST", "localhost")
            port = int(os.environ.get("METAL_REMOTE_PORT", "9123"))
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            print(f"Metal availability check failed: {e}")
            return False

    def _create_program(self, name: str, lib: bytes) -> RemoteMetalProgram:
        if DEBUG >= 3:
            print(f"\n=== RemoteMetalDevice creating program {name} ===")
            print(f"Binary size: {len(lib)} bytes")
            print(f"Binary starts with: {lib[:20].hex()}")
        try:
            program = RemoteMetalProgram(self, name, lib)
            if DEBUG >= 3: print(f"Created RemoteMetalProgram successfully")
            self.programs[name] = program
            return program
        except Exception as e:
            print(f"Error creating program: {e}")
            print(traceback.format_exc())
            raise

    def synchronize(self): pass

def patch_program_class():
    TinyMetalProgram.__call__ = MetalProgram.__call__
    TinyMetalProgram.__init__ = MetalProgram.__init__
    MetalProgram.__call__ = RemoteMetalProgram.__call__
    MetalProgram.__init__ = RemoteMetalProgram.__init__

def patch_device_class():
    original_method = Device.__class__._Device__get_canonicalized_item
    def new_get_canonicalized_item(self, ix: str) -> Any:
        if DEBUG >= 3: print(f"[DEV  ] Resolving device: {ix}")
        x = ix.split(":")[0].upper()
        if x == "METAL":
            host = os.environ.get("METAL_REMOTE_HOST", "localhost")
            port = int(os.environ.get("METAL_REMOTE_PORT", "9123"))
            ret = RemoteMetalDevice(ix, host=host, port=port)
            if DEBUG >= 3: print(f"[DEV  ] Using METAL device for {ix}")
            Device._opened_devices.add(ix)
            return ret
        return original_method(self, ix)
    Device.__class__._Device__get_canonicalized_item = new_get_canonicalized_item

# Apply patches and initialize
patch_device_class()
patch_program_class()

def init_metal_device():
    print("Initializing Metal device...")
    assert "METAL" in Device._devices
    if RemoteMetalDevice.available():
        print("Metal server is available, enabling METAL")
        os.environ["METAL"] = "1"
        Device._opened_devices.add("METAL")
        return True
    return False

def run_kernel_benchmark(num_iterations=1000, tensor_size=4):
    print("\n[BENCH] Creating test tensors...")
    a = Tensor.rand(tensor_size, requires_grad=False, device="METAL")
    b = Tensor.rand(tensor_size, requires_grad=False, device="METAL")
    print("[BENCH] Performing warmup...")
    result = (a + b).numpy()
    print(f"[BENCH] Warmup complete: {result}")
    print(f"[BENCH] Starting benchmark with {num_iterations} iterations...")
    start_time = time.time()
    for i in range(num_iterations):
        if i % 100 == 0:
            print(f"[BENCH] Iteration {i}/{num_iterations}")
        (a + b).numpy()
    end_time = time.time()
    total_time = end_time - start_time
    return num_iterations / total_time, (total_time / num_iterations) * 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Metal Runtime Test and Benchmark')
    parser.add_argument('--benchmark', action='store_true', help='Run kernel execution benchmark')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for benchmark')
    parser.add_argument('--tensor-size', type=int, default=4, help='Size of test tensors')
    args = parser.parse_args()

    os.environ["METAL_REMOTE_HOST"] = "localhost"
    os.environ["METAL_REMOTE_PORT"] = "9123"

    if not init_metal_device():
        print("\nFailed to initialize Metal device - is the server running?")
        os.abort()

    try:
        if args.benchmark:
            print("\nRunning kernel execution benchmark...")
            print(f"Parameters: iterations={args.iterations}, tensor_size={args.tensor_size}")
            executions_per_second, mean_time = run_kernel_benchmark(args.iterations, args.tensor_size)
            print("\nBenchmark Results:")
            print(f"Executions per second: {executions_per_second:.2f}")
            print(f"Mean execution time: {mean_time:.3f} ms")
        else:
            print("\nRunning basic operations test...")
            a = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=False, device="METAL")
            b = Tensor([0.5, 1.0, 1.5, 2.0], requires_grad=False, device="METAL")

            # Force realization and synchronization after each operation
            add_op = a + b
            add_result = add_op.numpy()

            mul_op = a * b
            mul_result = mul_op.numpy()

            sub_op = a - b
            sub_result = sub_op.numpy()

            div_op = a / b
            div_result = div_op.numpy()

            # Expected results
            expected_add = [1.5, 3.0, 4.5, 6.0]
            expected_mul = [0.5, 2.0, 4.5, 8.0]
            expected_sub = [0.5, 1.0, 1.5, 2.0]
            expected_div = [2.0, 2.0, 2.0, 2.0]

            # Print results with verification
            print("\nResults:")
            print(f"ADD: {add_result}")
            print(f"Expected: {expected_add}")
            print(f"Correct: {all(abs(a - b) < 1e-5 for a, b in zip(add_result, expected_add))}")

            print(f"\nMUL: {mul_result}")
            print(f"Expected: {expected_mul}")
            print(f"Correct: {all(abs(a - b) < 1e-5 for a, b in zip(mul_result, expected_mul))}")

            print(f"\nSUB: {sub_result}")
            print(f"Expected: {expected_sub}")
            print(f"Correct: {all(abs(a - b) < 1e-5 for a, b in zip(sub_result, expected_sub))}")

            print(f"\nDIV: {div_result}")
            print(f"Expected: {expected_div}")
            print(f"Correct: {all(abs(a - b) < 1e-5 for a, b in zip(div_result, expected_div))}")

            # Overall test result
            all_correct = (
                all(abs(a - b) < 1e-5 for a, b in zip(add_result, expected_add)) and
                all(abs(a - b) < 1e-5 for a, b in zip(mul_result, expected_mul)) and
                all(abs(a - b) < 1e-5 for a, b in zip(sub_result, expected_sub)) and
                all(abs(a - b) < 1e-5 for a, b in zip(div_result, expected_div))
            )

            print(f"\nAll tests {'PASSED' if all_correct else 'FAILED'}")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()