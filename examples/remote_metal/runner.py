import socket, json, struct, time
from typing import Dict, Any
from tinygrad.runtime.ops_metal import MetalDevice, MetalProgram, MetalBuffer, msg
import ctypes
from tinygrad.runtime.ops_metal import libobjc
import logging
import os

# Configure logging based on DEBUG environment variable
debug_level = int(os.getenv('DEBUG', '0'))
logging.basicConfig(
    level=logging.DEBUG if debug_level > 0 else logging.INFO,
    format='%(asctime)s - %(levelname)5s - %(message)s'
)

# Define objc types
objc_id = ctypes.c_void_p
def to_mv(buf, count) -> memoryview:
    return (ctypes.c_uint8 * count).from_address(buf).from_buffer()

class SimpleMetalRunner:
    def __init__(self, host="0.0.0.0", port=9123):
        self.host, self.port = host, port
        self.running = False
        self.server_socket = None
        print("Initializing Metal device...")
        self.device = MetalDevice("METAL")
        self._active_buffers = set()
        self._active_programs = set()

    def get_buffer_data(self, buf):
        """Extract raw data from a MetalBuffer using tinygrad's native methods."""
        try:
            # Use device allocator to get buffer contents
            view = self.device.allocator._as_buffer(buf)
            return view.tobytes().hex() if view is not None else None
        except Exception as e:
            if debug_level > 0: print(f"Failed to get buffer data: {e}")
            if debug_level >=3: logging.debug(f"Buffer access error: {e}")
            return None

    def set_buffer_data(self, buf, hex_data):
        """Update a MetalBuffer with new data using tinygrad's native methods."""
        if not hex_data: return
        try:
            raw_data = bytes.fromhex(hex_data)
            view = self.device.allocator._as_buffer(buf)
            if view is not None:
                view[:] = raw_data
                if debug_level > 3: print(f"Updated buffer with {len(raw_data)} bytes")
        except Exception as e:
            if debug_level > 0: print(f"Failed to update buffer: {e}")
            if debug_level >=3: logging.debug(f"Buffer update error: {e}")

    def format_kernel_info(self, name: str, kernel_time: float) -> str:
        kernel_time_us = kernel_time * 1_000_000
        return f"kernel {name:<50} {kernel_time_us:>12.2f} us"

    def execute_kernel(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            program = MetalProgram(self.device, request["name"], bytes.fromhex(request["kernel_obj"]))
            self._active_programs.add(program)

            # Create buffers and initialize them with provided data
            buffers = []
            for buffer_info in request["buffer_info"]:
                buf = self.device.allocator._alloc(buffer_info["size"], None)
                if buffer_info.get("offset", 0) > 0:
                    buf = self.device.allocator._offset(buf, buffer_info["size"], buffer_info["offset"])

                # Initialize buffer with provided data if any
                if buffer_info.get("data"):
                    self.set_buffer_data(buf, buffer_info["data"])

                buffers.append(buf)
                self._active_buffers.add(buf)

            # Execute kernel and measure timing
            start_time = time.time()
            if debug_level >=3: logging.debug(f"Executing kernel with {len(buffers)} buffers and values: {request['values']}")
            program(*buffers, vals=tuple(request["values"]), wait=request["wait"])
            kernel_time = time.time() - start_time
            if debug_level >=3: logging.debug(f"Kernel execution completed in {kernel_time:.6f} seconds")

            # Collect updated buffer data
            buffer_results = []
            for i, buf in enumerate(buffers):
                if debug_level >=3: logging.debug(f"Processing buffer {i}:")
                if debug_level >=3: logging.debug(f"  - Size: {buf.size}")
                if debug_level >=3: logging.debug(f"  - Offset: {buf.offset}")
                if debug_level >=3: logging.debug(f"  - Has realized: {hasattr(buf, 'realized')}")

                try:
                    data = self.get_buffer_data(buf)
                    if debug_level >=3: logging.debug(f"  - Got buffer data: {data[:50] if data else None}...")
                except Exception as e:
                    logging.error(f"  - Error getting buffer data: {e}")
                    data = None

                result = {
                    "size": buf.size,
                    "offset": buf.offset,
                    "data": data
                }
                buffer_results.append(result)

            kernel_info = self.format_kernel_info(request['name'], kernel_time)
            if debug_level >=2: logging.info(kernel_info)

            return {
                "status": "success",
                "buffers": buffer_results,
                "kernel_time": kernel_time
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if request.get("type") == "synchronize":
                logging.info("Synchronizing device...")
                self.device.synchronize()
                return {"status": "success"}
            elif request.get("type") == "kernel_execute":
                return self.execute_kernel(request)
            else:
                return {"status": "error", "error": f"Unknown request type: {request.get('type')}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cleanup_resources(self):
        if debug_level >=3: logging.debug("Starting resource cleanup")
        try:
            self._active_buffers.clear()
            self._active_programs.clear()
            if debug_level >=3: logging.debug("Resource cleanup completed")
        except Exception as e:
            logging.error(f"Error in cleanup_resources: {e}", exc_info=True)

    def handle_client(self, client_socket: socket.socket, addr: tuple):
        try:
            logging.info(f"Handling client connection from {addr}")
            while self.running:
                size_data = client_socket.recv(4)
                if not size_data: break

                request_size = struct.unpack("!I", size_data)[0]
                request_data = b""
                while len(request_data) < request_size:
                    chunk = client_socket.recv(min(request_size - len(request_data), 4096))
                    if not chunk: break
                    request_data += chunk

                if len(request_data) != request_size:
                    if debug_level >=3: logging.debug("Incomplete request received")
                    break

                request = json.loads(request_data.decode())
                start_time = time.time()

                result = self.handle_request(request)
                response = {**result, "execution_time": time.time() - start_time}

                response_bytes = json.dumps(response).encode()
                client_socket.sendall(struct.pack("!I", len(response_bytes)))
                client_socket.sendall(response_bytes)

        except Exception as e:
            logging.error(f"Error handling client {addr}: {e}")
        finally:
            logging.info(f"Cleaning up resources for client {addr}")
            self.cleanup_resources()
            client_socket.close()
            if debug_level >=3: logging.debug(f"Closed socket for client {addr}")

    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            logging.info(f"Metal runner listening on {self.host}:{self.port}")

            while self.running:
                try:
                    self.server_socket.settimeout(1.0)
                    try:
                        client_socket, addr = self.server_socket.accept()
                        # Handle client directly - no threading
                        self.handle_client(client_socket, addr)
                    except socket.timeout:
                        continue
                except Exception as e:
                    if self.running:
                        logging.error(f"Error accepting connection: {e}")
        except Exception as e:
            logging.error(f"Error starting server: {e}")
        finally:
            self.stop()

    def stop(self):
        logging.info("Shutting down Metal runner...")
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                logging.error(f"Error closing server socket: {e}")
        logging.info("Metal runner shutdown complete")

if __name__ == "__main__":
    runner = SimpleMetalRunner()
    try:
        runner.start()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt...")
    finally:
        runner.stop()