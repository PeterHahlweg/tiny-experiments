#!/usr/bin/env python
import subprocess
import json
import os
from datetime import datetime
import sys
import shlex

# Define the example commands to run
EXAMPLES = [
    "python examples/edge.py --use-test-image"
]

def run_tinygrad_example(command, optimization=False, reports_dir=None):
    """Run a TinyGrad example with or without optimization."""
    command_parts = shlex.split(command)
    script_path = command_parts[1]  # After 'python'
    name = os.path.basename(script_path).replace(".py", "")
    log_file = os.path.join(reports_dir, f"{name}_{'optimized' if optimization else 'baseline'}.log")

    env = os.environ.copy()
    env["DEBUG"] = "4"  # Enable detailed logging
    env["BEAM"] = "111" if optimization else "0"

    # Get absolute paths and set up working directory
    tools_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/tools
    project_dir = os.path.dirname(tools_dir)                # /path/to/project
    original_dir = os.getcwd()  # Save current working directory

    # Change to project directory before running
    os.chdir(project_dir)

    # Set up PYTHONPATH to include the project directory
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        env['PYTHONPATH'] = f"{project_dir}:{current_pythonpath}"
    else:
        env['PYTHONPATH'] = project_dir

    try:
        # Check if tinygrad is installed
        import importlib.util
        tinygrad_spec = importlib.util.find_spec("tinygrad")
        if tinygrad_spec is None:
            potential_tinygrad_dir = os.path.join(project_dir, "tinygrad")
            if not os.path.exists(potential_tinygrad_dir):
                raise ImportError(f"tinygrad module not found in Python path and not found at {potential_tinygrad_dir}")

        # Replace 'python' with actual Python executable path
        command_parts[0] = sys.executable

        with open(log_file, "w") as f:
            result = subprocess.run(
                command_parts,
                env=env,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

        # Verify the log file has content
        if os.path.getsize(log_file) == 0:
            print(f"Warning: Log file {log_file} is empty!")
        else:
            print(f"Log file created: {log_file}")

        return log_file

    except ImportError as e:
        print(f"Error: {str(e)}")
        print("Please ensure tinygrad is installed or the repository is properly cloned.")
        print("You can install tinygrad via:")
        print("  pip install tinygrad")
        print("Or clone the repository and ensure you're in the correct directory.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error running example: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise
    finally:
        # Always restore the original working directory
        os.chdir(original_dir)

def analyze_log(log_file):
    """Run the analyzer on the log file and return the JSON results."""
    json_file = f"{log_file}.json"
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"\nAnalyzing log file: {log_file}")

    # Ensure we're in the project directory when running the analyzer
    original_dir = os.getcwd()
    try:
        os.chdir(project_dir)
        result = subprocess.run(
            ["python3", os.path.join("tools", "analyse.py"), log_file],
            stdout=open(json_file, 'w'),
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        if not os.path.exists(json_file):
            print(f"Warning: JSON file {json_file} was not created by analyzer")
            return {"total_runtime_us": 0, "kernels": []}

        with open(json_file) as f:
            return json.load(f)

    except subprocess.CalledProcessError as e:
        print(f"Error running analyzer: {e.stderr}")
        raise
    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error during analysis: {str(e)}")
        raise
    finally:
        os.chdir(original_dir)

def calculate_performance_metrics(baseline_data, optimized_data):
    """Calculate performance comparison metrics."""
    baseline_time = baseline_data.get("total_runtime_us", 0)
    optimized_time = optimized_data.get("total_runtime_us", 0)

    # Calculate total memory and GFLOPS
    def sum_kernel_metrics(data, metric):
        return sum(k.get("metrics", {}).get(metric, 0) for k in data.get("kernels", []))

    baseline_memory = sum_kernel_metrics(baseline_data, "memory_GB")
    optimized_memory = sum_kernel_metrics(optimized_data, "memory_GB")
    baseline_gflops = sum_kernel_metrics(baseline_data, "gflops")
    optimized_gflops = sum_kernel_metrics(optimized_data, "gflops")

    speedup = baseline_time / optimized_time if optimized_time > 0 else 0
    time_reduction = baseline_time - optimized_time
    improvement_percent = (time_reduction / baseline_time * 100) if baseline_time > 0 else 0
    memory_impact = f"{optimized_memory:.3f}GB vs {baseline_memory:.3f}GB"
    gflops_improvement = f"{optimized_gflops:.2f} vs {baseline_gflops:.2f}"

    return {
        "speedup": speedup,
        "time_reduction": time_reduction,
        "improvement_percent": improvement_percent,
        "memory_impact": memory_impact,
        "gflops_comparison": gflops_improvement,
        "total_gflops_optimized": optimized_gflops,
        "total_gflops_baseline": baseline_gflops
    }

def format_kernel_table(kernel_data):
    """Format kernel data into markdown table rows."""
    rows = []
    total_time_us = 0

    for kernel in kernel_data.get("kernels", []):
        # Skip memory operations
        if "copy" in kernel.get('kernel_name', '').lower():
            continue

        metrics = kernel.get("metrics", {})
        kernel_name = kernel.get('kernel_name', '')
        timing_us = metrics.get('timing_us', 0)
        total_time_us += timing_us

        # Extract shape from kernel name if available
        shape = "-"
        if "_" in kernel_name and not kernel_name.startswith("copy"):
            shape = kernel_name  # The kernel name itself contains the shape information
            kernel_name = "Metal kernel"  # Simplified name

        rows.append(f"| {kernel_name} | {shape} | "
                   f"{metrics.get('memory_GB', 0):.3f} | {timing_us:.2f} | "
                   f"{metrics.get('gflops', 0):.2f} | {metrics.get('bandwidth', 0) or 0:.2f} | "
                   f"{kernel.get('backend', '')} |")

    # Add total runtime in milliseconds
    total_time_ms = total_time_us / 1000.0
    rows.append(f"| **Total Time** | | | {total_time_ms:.2f} ms | | | |")
    return rows

def format_memory_operations(data):
    """Format memory operations into markdown table rows."""
    rows = []
    memory_kernels = [k for k in data.get("kernels", [])
                     if "copy" in k.get("kernel_name", "").lower()]

    for kernel in memory_kernels:
        metrics = kernel.get("metrics", {})
        # Extract just the transfer direction (e.g., "METAL <- NPY") from the kernel name
        kernel_name = kernel.get('kernel_name', '')
        if ", " in kernel_name:
            transfer_direction = kernel_name.split(", ", 1)[1]  # Get everything after the first comma
        else:
            transfer_direction = kernel_name  # Fallback if no comma found

        rows.append(f"| {transfer_direction} | {metrics.get('memory_GB', 0):.3f} | "
                   f"{metrics.get('timing_us', 0):.2f} |")

    # Add total for memory operations
    total_time = sum(float(k.get("metrics", {}).get("timing_us", 0)) for k in memory_kernels)
    total_memory = sum(float(k.get("metrics", {}).get("memory_GB", 0)) for k in memory_kernels)
    total_memory = sum(float(k.get("metrics", {}).get("memory_GB", 0)) for k in memory_kernels)
    rows.append(f"| **Total** | {total_memory:.3f} | {total_time:.2f} |")
    return rows

def generate_markdown_report(command, baseline_data, optimized_data, metrics):
    """Generate the markdown report comparing both runs."""
    command_parts = command.split()
    example_name = os.path.basename(command_parts[1])

    report = [
        f"# Kernel Optimisation Report - {example_name}\n",
        f"Program: `{command}`\n",
        "## 1. Non-Optimized Compute Kernels\n",
        "| Kernel ID | Memory (GB) | Time (μs) | GFLOPS | Bandwidth (GB/s) | Backend |",
        "|-----------|-------------|-----------|---------|-----------------|---------|"
    ]

    # Modified format_kernel_table function inline
    def format_kernel_table(kernel_data):
        rows = []
        total_time_us = 0
        total_memory = 0
        total_gflops = 0
        total_bandwidth = 0

        for kernel in kernel_data.get("kernels", []):
            if "copy" in kernel.get('kernel_name', '').lower():
                continue

            metrics = kernel.get("metrics", {})
            kernel_name = kernel.get('kernel_name', '')
            timing_us = metrics.get('timing_us', 0)
            memory_gb = metrics.get('memory_GB', 0)
            gflops = metrics.get('gflops', 0)
            bandwidth = metrics.get('bandwidth', 0) or 0

            total_time_us += timing_us
            total_memory += memory_gb
            total_gflops += gflops
            total_bandwidth += bandwidth

            # Use shape as the Kernel ID
            if "_" in kernel_name and not kernel_name.startswith("copy"):
                shape = kernel_name
                rows.append(f"| {shape} | "
                       f"{memory_gb:.3f} | {timing_us:.2f} | "
                       f"{gflops:.2f} | {bandwidth:.2f} | "
                       f"{kernel.get('backend', '')} |")

        # Add totals row with all columns
        total_time_ms = total_time_us / 1000.0
        rows.append(f"| **Total** | {total_memory:.3f} | {total_time_us:.2f} | {total_gflops:.2f} | {total_bandwidth:.2f} | |")
        return rows

    report.extend(format_kernel_table(baseline_data))

    report.extend([
        "\n## 2. BEAM Optimized Compute Kernels\n",
        "| Kernel ID | Memory (GB) | Time (μs) | GFLOPS | Bandwidth (GB/s) | Backend |",
        "|-----------|-------------|-----------|---------|-----------------|---------|"
    ])

    report.extend(format_kernel_table(optimized_data))

    report.extend([
        "\n## 3. Memory Transfer Operations\n",
        "| Transfer Direction | Memory (GB) | Duration (μs) |",
        "|-------------------|-------------|---------------|"
    ])

    report.extend(format_memory_operations(optimized_data))

    # Calculate additional metrics
    efficiency = (metrics['total_gflops_optimized'] / metrics['total_gflops_baseline'] - 1) * 100
    time_reduction_ms = metrics['time_reduction'] / 1000.0  # Convert μs to ms

    report.extend([
        "\n## 4. Performance Analysis\n",
        "| Metric | Value | Notes |",
        "|--------|-------|-------|",
        f"| Speed-up Factor | {metrics['speedup']:.2f}x | Total execution time improvement |",
        f"| Time Reduction | {time_reduction_ms:.2f}ms | Absolute time saved |",
        f"| Improvement | {metrics['improvement_percent']:.1f}% | Reduction in execution time |",
        f"| Memory Impact | {metrics['memory_impact']} | Memory footprint comparison |",
        f"| GFLOPS | {metrics['gflops_comparison']} | Computational throughput |",
        f"| GFLOPS Improvement | {efficiency:.1f}% | Compute efficiency gain |"
    ])

    return "\n".join(report)

def main():
    """Main function to run the analysis and generate reports for all examples."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    reports_dir = f"performance_reports_{timestamp}"
    os.makedirs(reports_dir, exist_ok=True)

    for command in EXAMPLES:
        print(f"\nAnalyzing {command}...")

        # Run baseline
        baseline_log = run_tinygrad_example(command, optimization=False, reports_dir=reports_dir)
        baseline_data = analyze_log(baseline_log)

        # Run optimized
        optimized_log = run_tinygrad_example(command, optimization=True, reports_dir=reports_dir)
        optimized_data = analyze_log(optimized_log)

        # Calculate performance metrics
        metrics = calculate_performance_metrics(baseline_data, optimized_data)

        # Generate and save report
        report = generate_markdown_report(command, baseline_data, optimized_data, metrics)
        example_name = os.path.basename(command.split()[1]).replace(".py", "")
        report_file = os.path.join(reports_dir, f"{example_name}_analysis.md")

        with open(report_file, "w") as f:
            f.write(report)

        print(f"Report saved to {report_file}")

        # Don't cleanup files anymore - keep them in the reports directory for reference
        print(f"All files saved in: {reports_dir}")

if __name__ == "__main__":
    main()