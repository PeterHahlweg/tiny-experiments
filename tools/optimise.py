#!/usr/bin/env python
import subprocess
import json
import os
from datetime import datetime
import sys
import shlex
from typing import List, Dict
import statistics

# Define the programs to run
PROGRAMS = [
    "python examples/edge.py --use-test-image"
]

def run_tinygrad_program(command: str, optimization: bool = False, reports_dir: str = None, run_index: int = 0, beam_value: str = "0") -> str:
    """Run a TinyGrad program with or without optimization."""
    command_parts = shlex.split(command)
    script_path = command_parts[1]  # After 'python'
    name = os.path.basename(script_path).replace(".py", "")
    suffix = 'optimized' if optimization else 'baseline'
    log_file = os.path.join(reports_dir, f"{name}_{suffix}_run{run_index}.log")

    env = os.environ.copy()
    env["DEBUG"] = "4"  # Enable detailed logging
    env["BEAM"] = beam_value if optimization else "0"  # Use passed beam_value

    # Get absolute paths and set up working directory
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(tools_dir)
    original_dir = os.getcwd()

    os.chdir(project_dir)

    # Set up PYTHONPATH
    current_pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{project_dir}:{current_pythonpath}" if current_pythonpath else project_dir

    try:
        # Check if tinygrad is installed
        import importlib.util
        tinygrad_spec = importlib.util.find_spec("tinygrad")
        if tinygrad_spec is None:
            potential_tinygrad_dir = os.path.join(project_dir, "tinygrad")
            if not os.path.exists(potential_tinygrad_dir):
                raise ImportError(f"tinygrad module not found in Python path and not found at {potential_tinygrad_dir}")

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

        if os.path.getsize(log_file) == 0:
            print(f"Warning: Log file {log_file} is empty!")
        else:
            print(f"Log file created: {log_file}")

        return log_file

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise
    finally:
        os.chdir(original_dir)

def analyze_log(log_file: str) -> Dict:
    """Run the analyzer on the log file and return the JSON results."""
    json_file = f"{log_file}.json"
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"\nAnalyzing log file: {log_file}")

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

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise
    finally:
        os.chdir(original_dir)

def get_total_runtime(data: Dict) -> float:
    """Calculate total runtime from all kernels in the analysis data."""
    return sum(kernel.get("metrics", {}).get("timing_us", 0) for kernel in data.get("kernels", []))

def get_total_runtime_with_transfers(data: Dict) -> tuple[float, float, float]:
    """Calculate total runtime including both compute and memory transfer operations."""
    compute_time = 0
    transfer_time = 0

    for kernel in data.get("kernels", []):
        timing = kernel.get("metrics", {}).get("timing_us", 0)
        if "copy" in kernel.get("kernel_name", "").lower():
            transfer_time += timing
        else:
            compute_time += timing

    return compute_time, transfer_time, compute_time + transfer_time

def get_best_run(runs: List[Dict]) -> Dict:
    """Determine the fastest run from multiple executions based on kernel timings."""
    return min(runs, key=get_total_runtime)

def calculate_performance_metrics(baseline_runs: List[Dict], optimized_runs: List[Dict]) -> Dict:
    """Calculate performance comparison metrics."""
    best_baseline = get_best_run(baseline_runs)
    best_optimized = get_best_run(optimized_runs)

    baseline_time = get_total_runtime(best_baseline)
    optimized_time = get_total_runtime(best_optimized)

    # Calculate statistics for all runs
    baseline_times = [get_total_runtime(run) for run in baseline_runs]
    optimized_times = [get_total_runtime(run) for run in optimized_runs]

    def sum_kernel_metrics(data: Dict, metric: str) -> float:
        return sum(k.get("metrics", {}).get(metric, 0) for k in data.get("kernels", []))

    baseline_memory = sum_kernel_metrics(best_baseline, "memory_GB")
    optimized_memory = sum_kernel_metrics(best_optimized, "memory_GB")
    baseline_gflops = sum_kernel_metrics(best_baseline, "gflops")
    optimized_gflops = sum_kernel_metrics(best_optimized, "gflops")

    return {
        "speedup": baseline_time / optimized_time if optimized_time > 0 else 0,
        "time_reduction": baseline_time - optimized_time,
        "improvement_percent": (baseline_time - optimized_time) / baseline_time * 100 if baseline_time > 0 else 0,
        "memory_impact": f"{optimized_memory:.3f}GB vs {baseline_memory:.3f}GB",
        "gflops_comparison": f"{optimized_gflops:.2f} vs {baseline_gflops:.2f}",
        "total_gflops_optimized": optimized_gflops,
        "total_gflops_baseline": baseline_gflops,
        "baseline_stats": {
            "mean": statistics.mean(baseline_times),
            "std": statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0,
            "min": min(baseline_times),
            "max": max(baseline_times)
        },
        "optimized_stats": {
            "mean": statistics.mean(optimized_times),
            "std": statistics.stdev(optimized_times) if len(optimized_times) > 1 else 0,
            "min": min(optimized_times),
            "max": max(optimized_times)
        }
    }

def format_kernel_table(kernel_data: Dict) -> List[str]:
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

def format_memory_operations(data: Dict) -> List[str]:
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
    rows.append(f"| **Total** | {total_memory:.3f} | {total_time:.2f} |")
    return rows

def generate_markdown_report(command: str, baseline_runs: List[Dict], optimized_runs: List[Dict],
                           metrics: Dict, beam_value: str) -> str:
    """Generate the markdown report comparing both runs."""
    best_baseline = get_best_run(baseline_runs)
    best_optimized = get_best_run(optimized_runs)

    command_parts = command.split()
    example_name = os.path.basename(command_parts[1])

    report = [
        f"# Kernel Optimisation Report - {example_name}\n",
        f"Program: `{command}`\n",
        "\n## 1. Non-Optimized Compute Kernels\n",
        "| Kernel | Shape | Memory (GB) | Time (μs) | GFLOPS | Bandwidth (GB/s) | Backend |",
        "|---------|-------|-------------|------------|---------|-----------------|----------|"
    ]

    report.extend(format_kernel_table(best_baseline))

    report.extend([
        f"\n## 2. Optimized Compute Kernels - BEAM {beam_value}\n",  # Use passed beam_value
        "| Kernel | Shape | Memory (GB) | Time (μs) | GFLOPS | Bandwidth (GB/s) | Backend |",
        "|---------|-------|-------------|------------|---------|-----------------|----------|"
    ])

    report.extend(format_kernel_table(best_optimized))

    report.extend([
        "\n## 3. Memory Transfer Operations\n",
        "| Transfer Direction | Memory (GB) | Duration (μs) |",
        "|-------------------|-------------|---------------|"
    ])

    report.extend(format_memory_operations(best_optimized))

    # Calculate additional metrics
    efficiency = (metrics['total_gflops_optimized'] / metrics['total_gflops_baseline'] - 1) * 100
    time_reduction_ms = metrics['time_reduction'] / 1000.0  # Convert μs to ms

    # Calculate complete runtimes including transfers
    baseline_compute, baseline_transfer, baseline_total = get_total_runtime_with_transfers(best_baseline)
    optimized_compute, optimized_transfer, optimized_total = get_total_runtime_with_transfers(best_optimized)

    # Convert to milliseconds for better readability
    baseline_total_ms = baseline_total / 1000
    optimized_total_ms = optimized_total / 1000

    report.extend([
        "\n## 4. Performance Analysis\n",
        "| Metric | Value | Notes |",
        "|--------|-------|-------|",
        f"| Complete Runtime | {optimized_total_ms:.2f}ms | Includes compute ({optimized_compute/1000:.2f}ms) and transfers ({optimized_transfer/1000:.2f}ms) |",
        f"| Speed-up Factor | {metrics['speedup']:.2f}x | Total execution time improvement |",
        f"| Time Reduction | {time_reduction_ms:.2f}ms | Absolute time saved |",
        f"| Improvement | {metrics['improvement_percent']:.1f}% | Reduction in execution time |",
        f"| Memory Impact | {metrics['memory_impact']} | Memory footprint comparison |",
        f"| GFLOPS | {metrics['gflops_comparison']} | Computational throughput |",
        f"| GFLOPS Improvement | {efficiency:.1f}% | Compute efficiency gain |",
    ])

    report.extend([
        "\n## Execution Statistics\n",
        "| Metric | Baseline | Optimized |",
        "|--------|----------|-----------|",
        f"| Best Runtime (μs) | {metrics['baseline_stats']['min']:.2f} | {metrics['optimized_stats']['min']:.2f} |",
        f"| Mean Runtime (μs) | {metrics['baseline_stats']['mean']:.2f} | {metrics['optimized_stats']['mean']:.2f} |",
        f"| Std Dev (μs) | {metrics['baseline_stats']['std']:.2f} | {metrics['optimized_stats']['std']:.2f} |",
        f"| Worst Runtime (μs) | {metrics['baseline_stats']['max']:.2f} | {metrics['optimized_stats']['max']:.2f} |",
    ])

    # Add section for kernel code
    report.extend([
        "\n## 5. Kernel Source Code\n"
    ])

    # Function to extract and format kernel code
    def format_kernel_code(kernel_data: Dict) -> List[str]:
        kernel_sections = []
        for kernel in kernel_data.get("kernels", []):
            # Skip memory operations
            if "copy" in kernel.get('kernel_name', '').lower():
                continue

            kernel_name = kernel.get('kernel_name', '')
            code = kernel.get('code')
            if code:
                kernel_sections.extend([
                    f"\n### Kernel: {kernel_name}\n",
                    "```c",
                    code,
                    "```\n"
                ])
        return kernel_sections

    # Add kernel code from both baseline and optimized runs
    report.extend(format_kernel_code(best_baseline))
    # If the kernels are different in optimized version, add those too
    optimized_kernels = {k.get('kernel_name'): k for k in best_optimized.get("kernels", [])
                        if "copy" not in k.get('kernel_name', '').lower()}
    baseline_kernels = {k.get('kernel_name'): k for k in best_baseline.get("kernels", [])
                       if "copy" not in k.get('kernel_name', '').lower()}

    # Only add optimized kernels if they're different from baseline
    if optimized_kernels != baseline_kernels:
        report.extend(["\n### Optimized Kernels"])
        report.extend(format_kernel_code(best_optimized))

    return "\n".join(report)

def main():
    """Main function to run the analysis with multiple iterations."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    reports_dir = f"performance_reports_{timestamp}"
    os.makedirs(reports_dir, exist_ok=True)

    NUM_RUNS = 10  # Number of runs for each phase
    BEAM_VALUE = os.environ.get("BEAM", "3")  # Get BEAM value from environment

    for command in PROGRAMS:
        print(f"\nAnalyzing {command}...")

        # Phase 1: Run unoptimized program 10 times
        baseline_runs = []
        print("\nPhase 1: Running baseline tests...")
        for i in range(NUM_RUNS):
            print(f"Baseline run {i+1}/{NUM_RUNS}")
            baseline_log = run_tinygrad_program(command, optimization=False, reports_dir=reports_dir, run_index=i, beam_value=BEAM_VALUE)
            baseline_runs.append(analyze_log(baseline_log))

        # Phase 2: Run optimization step once
        print("\nPhase 2: Running optimization step...")
        optimization_log = run_tinygrad_program(command, optimization=True, reports_dir=reports_dir, run_index="opt", beam_value=BEAM_VALUE)
        _ = analyze_log(optimization_log)  # We analyze but don't need to store the result

        # Phase 3: Run optimized version 10 times
        optimized_runs = []
        print("\nPhase 3: Running optimized tests...")
        for i in range(NUM_RUNS):
            print(f"Optimized run {i+1}/{NUM_RUNS}")
            optimized_log = run_tinygrad_program(command, optimization=True, reports_dir=reports_dir, run_index=i, beam_value=BEAM_VALUE)
            optimized_runs.append(analyze_log(optimized_log))

        # Calculate metrics using all runs
        metrics = calculate_performance_metrics(baseline_runs, optimized_runs)

        # Generate and save report
        report = generate_markdown_report(command, baseline_runs, optimized_runs, metrics, BEAM_VALUE)  # Pass BEAM_VALUE
        example_name = os.path.basename(command.split()[1]).replace(".py", "")
        report_file = os.path.join(reports_dir, f"{example_name}_analysis.md")

        with open(report_file, "w") as f:
            f.write(report)

        print(f"\nReport saved to {report_file}")
        print(f"All files saved in: {reports_dir}")

if __name__ == "__main__":
    main()