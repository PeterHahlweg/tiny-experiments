import re
import sys
import json
import argparse
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class KernelInfo:
    name: str
    device: str
    runtime: float  # in microseconds
    memory: float  # in GB
    gflops: float
    bandwidth: tuple[float, float]  # GB/s (read|write)
    operations: List[str]
    parameters: Optional[str] = None
    kernel_code: Optional[str] = None

def remove_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_tinygrad_log(log_content: str) -> List[KernelInfo]:
    kernels = []
    current_kernel = None
    lines = [line for line in log_content.split('\n') if line.strip()]

    # Updated pattern to be more flexible
    kernel_pattern = r'\*\*\* (\w+)\s+\d+\s+([^,]+(?:,[\s\w<>-]+)?)\s+arg\s+(\d+)\s+mem\s+([\d.]+)\s+GB\s+tm\s+([\d.]+)us/\s*([\d.]+)ms\s*\(\s*([\d.]+)\s+GFLOPS\s+([\d.]+)\|([\d.]+)\s+GB/s\)(?:\s+\[(.*?)\])?'

    for i, line in enumerate(lines):
        # Remove ANSI color codes before processing
        clean_line = remove_ansi_codes(line.strip())

        # Try to match kernel information line
        kernel_match = re.match(kernel_pattern, clean_line)

        if kernel_match:
            # If we were processing a previous kernel, store its code
            if current_kernel:
                kernels.append(current_kernel)

            # Parse kernel information
            device = kernel_match.group(1)
            name = kernel_match.group(2).strip()
            memory = float(kernel_match.group(4))
            # Use the ms value for runtime instead of us
            runtime = float(kernel_match.group(6)) * 1000  # Convert ms to us
            gflops = float(kernel_match.group(7))
            bandwidth = (float(kernel_match.group(8)), float(kernel_match.group(9)))
            operations = []
            if kernel_match.group(10):  # Operations are optional
                operations = [op.strip() for op in kernel_match.group(10).split(',') if op.strip()]

            # Extract kernel shape parameters from name
            parameters = None
            if '_' in name:
                parameters = name.split(',')[0].strip()  # Get the r_XXX part before any comma

            current_kernel = KernelInfo(
                name=name,
                device=device,
                runtime=runtime,
                memory=memory,
                gflops=gflops,
                bandwidth=bandwidth,
                operations=operations,
                parameters=parameters
            )

            # Look ahead for kernel code
            code_buffer = []
            in_code = False
            for next_line in lines[i+1:]:
                clean_next_line = remove_ansi_codes(next_line)

                # Break if we hit the next kernel
                if clean_next_line.startswith('***'):
                    break

                # Start collecting code at metal_stdlib
                if '#include <metal_stdlib>' in clean_next_line:
                    in_code = True
                # Skip UOp sections
                elif clean_next_line.startswith('UOp('):
                    in_code = False
                    continue
                # Skip Opt sections
                elif clean_next_line.startswith('[Opt('):
                    in_code = False
                    continue

                if in_code:
                    code_buffer.append(clean_next_line)

            if code_buffer:
                current_kernel.kernel_code = '\n'.join(code_buffer)

    # Don't forget the last kernel
    if current_kernel:
        kernels.append(current_kernel)

    return kernels

def generate_kernel_summary(kernels: List[KernelInfo]):
    total_runtime = sum(k.runtime for k in kernels)
    total_gflops = sum(k.gflops for k in kernels)

    # Group kernels by device
    by_device = {}
    for k in kernels:
        by_device.setdefault(k.device, []).append(k)

    # Create device summaries
    device_summaries = {}
    for device, device_kernels in by_device.items():
        device_summaries[device] = {
            "count": len(device_kernels),
            "total_runtime": round(sum(k.runtime for k in device_kernels), 2),
            "total_gflops": round(sum(k.gflops for k in device_kernels), 2)
        }

    # Create detailed kernel info
    kernel_details = []
    for i, kernel in enumerate(kernels, 1):
        kernel_info = {
            "id": i,
            "name": kernel.name,
            "device": kernel.device,
            "runtime": round(kernel.runtime, 2),
            "memory": round(kernel.memory, 2),
            "gflops": round(kernel.gflops, 2),
            "bandwidth": {
                "read": round(kernel.bandwidth[0], 1),
                "write": round(kernel.bandwidth[1], 1)
            }
        }
        
        if kernel.operations:
            kernel_info["operations"] = kernel.operations
        if kernel.parameters:
            kernel_info["parameters"] = kernel.parameters
        if kernel.kernel_code:
            kernel_info["kernel_code"] = kernel.kernel_code
            
        kernel_details.append(kernel_info)

    return {
        "summary": {
            "kernel_count": len(kernels),
            "total_runtime": round(total_runtime, 2),
            "total_gflops": round(total_gflops, 2),
            "average_gflops_per_kernel": round(total_gflops/len(kernels), 2)
        },
        "device_summaries": device_summaries,
        "kernels": kernel_details
    }

def main():
    parser = argparse.ArgumentParser(description='Parse and analyze tinygrad logs')
    parser.add_argument('log_file', type=str, help='Path to the tinygrad log file')

    args = parser.parse_args()

    try:
        # Read log file
        with open(args.log_file, 'r') as f:
            log_content = f.read()

        # Parse and output JSON summary
        kernels = parse_tinygrad_log(log_content)
        summary = generate_kernel_summary(kernels)
        print(json.dumps(summary, indent=2))

    except FileNotFoundError:
        print(f"Error: Could not find log file '{args.log_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing log file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
