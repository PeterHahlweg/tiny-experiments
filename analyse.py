import re
import sys
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
            runtime = float(kernel_match.group(5))
            gflops = float(kernel_match.group(7))
            bandwidth = (float(kernel_match.group(8)), float(kernel_match.group(9)))
            operations = []
            if kernel_match.group(10):  # Operations are optional
                operations = [op.strip() for op in kernel_match.group(10).split(',') if op.strip()]

            current_kernel = KernelInfo(
                name=name,
                device=device,
                runtime=runtime,
                memory=memory,
                gflops=gflops,
                bandwidth=bandwidth,
                operations=operations
            )

            # Look ahead for kernel parameters and code
            code_buffer = []
            in_code = False
            for next_line in lines[i+1:]:
                clean_next_line = remove_ansi_codes(next_line)

                # Break if we hit the next kernel
                if clean_next_line.startswith('***'):
                    break

                # Capture beam/hc parameters
                if clean_next_line.strip().startswith(('beam', 'hc')):
                    current_kernel.parameters = clean_next_line.strip()
                    continue

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

def print_kernel_summary(kernels: List[KernelInfo]):
    print(f"\nFound {len(kernels)} kernels\n")

    total_runtime = sum(k.runtime for k in kernels)
    total_gflops = sum(k.gflops for k in kernels)

    # Print overall summary
    print("Overall Summary:")
    print(f"Total Runtime: {total_runtime:.2f} µs")
    print(f"Total GFLOPS: {total_gflops:.2f}")
    print(f"Average GFLOPS per kernel: {total_gflops/len(kernels):.2f}\n")

    # Group kernels by device
    by_device = {}
    for k in kernels:
        by_device.setdefault(k.device, []).append(k)

    for device, device_kernels in by_device.items():
        print(f"\n{device} Kernels Summary:")
        print(f"Count: {len(device_kernels)}")
        print(f"Total Runtime: {sum(k.runtime for k in device_kernels):.2f} µs")
        print(f"Total GFLOPS: {sum(k.gflops for k in device_kernels):.2f}")

    print("\nDetailed Kernel Information:")
    print("=" * 80)

    for i, kernel in enumerate(kernels, 1):
        print(f"\nKernel {i}: {kernel.name}")
        print(f"{'='*80}")
        print(f"Device: {kernel.device}")
        print(f"Runtime: {kernel.runtime:.2f} µs")
        print(f"Memory: {kernel.memory:.2f} GB")
        print(f"Performance: {kernel.gflops:.2f} GFLOPS")
        print(f"Bandwidth (Read|Write): {kernel.bandwidth[0]:.1f}|{kernel.bandwidth[1]:.1f} GB/s")

        if kernel.operations:
            print(f"Operations: {', '.join(kernel.operations)}")

        if kernel.parameters:
            print("\nParameters:")
            print(kernel.parameters)

        if kernel.kernel_code:
            print("\nKernel Code:")
            print(kernel.kernel_code)

        print("\n" + "-"*80)

def main():
    parser = argparse.ArgumentParser(description='Parse and analyze tinygrad logs')
    parser.add_argument('log_file', type=str, help='Path to the tinygrad log file')

    args = parser.parse_args()

    try:
        # Read log file
        with open(args.log_file, 'r') as f:
            log_content = f.read()

        # Parse and print summary
        kernels = parse_tinygrad_log(log_content)
        print_kernel_summary(kernels)

    except FileNotFoundError:
        print(f"Error: Could not find log file '{args.log_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing log file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()