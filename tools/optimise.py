#!/usr/bin/env python
import subprocess
import json
import os
from datetime import datetime
import sys
import shlex
from typing import List, Dict
import statistics
import argparse

PROGRAMS = [
    "python examples/edge_detection/edge.py --use-test-image"
]

def run_tinygrad_program(command: str, optimization: bool = False, output_dir: str = None, run_index: int = 0, beam_value: str = "0") -> Dict:
    command_parts = shlex.split(command)
    script_path = command_parts[1]
    name = os.path.basename(script_path).replace(".py", "")

    env = os.environ.copy()
    env["DEBUG"] = "2"
    env["BEAM"] = beam_value if optimization else "0"

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(tools_dir)
    original_dir = os.getcwd()

    if output_dir:
        analysis_file = os.path.join(output_dir, f"{name}_{'optimized' if optimization else 'baseline'}_run{run_index}.json")
        command_parts.extend(["--analysis-output-file", analysis_file])
        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)

    os.chdir(project_dir)
    env['PYTHONPATH'] = f"{project_dir}:{env.get('PYTHONPATH', '')}"

    try:
        import importlib.util
        if importlib.util.find_spec("tinygrad") is None and not os.path.exists(os.path.join(project_dir, "tinygrad")):
            raise ImportError("tinygrad module not found")

        command_parts[0] = sys.executable

        # During optimization (Phase 2), show real-time output
        if optimization and run_index == 0:
            print(f"\nExecuting: {' '.join(command_parts)}")
            result = subprocess.run(command_parts, env=env, check=True)
        else:
            # For other phases, capture output silently
            result = subprocess.run(
                command_parts,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

        if os.path.exists(analysis_file):
            with open(analysis_file) as f:
                return json.load(f)
        else:
            return {"total_runtime_us": 0, "kernels": []}

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise
    finally:
        os.chdir(original_dir)

def get_total_runtime(data: Dict) -> float:
    return sum(kernel.get("metrics", {}).get("timing_us", 0) for kernel in data.get("kernels", []))

def get_total_runtime_with_transfers(data: Dict) -> tuple[float, float, float]:
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
    return min(runs, key=get_total_runtime)

def calculate_performance_metrics(baseline_runs: List[Dict], optimized_runs: List[Dict]) -> Dict:
    best_baseline = get_best_run(baseline_runs)
    best_optimized = get_best_run(optimized_runs)

    baseline_time = get_total_runtime(best_baseline)
    optimized_time = get_total_runtime(best_optimized)

    baseline_times = [get_total_runtime(run) for run in baseline_runs]
    optimized_times = [get_total_runtime(run) for run in optimized_runs]

    def sum_kernel_metrics(data: Dict, metric: str) -> float:
        return sum(float(k.get("metrics", {}).get(metric, 0) or 0) for k in data.get("kernels", []))

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
    rows = []
    total_time_us = 0

    for kernel in kernel_data.get("kernels", []):
        if "copy" in kernel.get('kernel_name', '').lower():
            continue

        metrics = kernel.get("metrics", {})
        kernel_name = kernel.get('kernel_name', '')
        timing_us = metrics.get('timing_us', 0)
        total_time_us += timing_us

        shape = "-"
        if "_" in kernel_name and not kernel_name.startswith("copy"):
            shape = kernel_name
            kernel_name = "Metal kernel"

        rows.append(f"| {kernel_name} | {shape} | "
                   f"{metrics.get('memory_GB', 0):.3f} | {timing_us:.2f} | "
                   f"{metrics.get('gflops', 0):.2f} | {float(metrics.get('bandwidth', 0) or 0):.2f} | "
                   f"{kernel.get('backend', '')} |")

    total_time_ms = total_time_us / 1000.0
    rows.append(f"| **Total Time** | | | {total_time_ms:.2f} ms | | | |")
    return rows

def format_memory_operations(data: Dict) -> List[str]:
    rows = []
    memory_kernels = [k for k in data.get("kernels", [])
                     if "copy" in k.get("kernel_name", "").lower()]

    for kernel in memory_kernels:
        metrics = kernel.get("metrics", {})
        kernel_name = kernel.get('kernel_name', '')
        transfer_direction = kernel_name.split(", ", 1)[1] if ", " in kernel_name else kernel_name

        rows.append(f"| {transfer_direction} | {metrics.get('memory_GB', 0):.3f} | "
                   f"{metrics.get('timing_us', 0):.2f} |")

    total_time = sum(float(k.get("metrics", {}).get("timing_us", 0)) for k in memory_kernels)
    total_memory = sum(float(k.get("metrics", {}).get("memory_GB", 0)) for k in memory_kernels)
    rows.append(f"| **Total** | {total_memory:.3f} | {total_time:.2f} |")
    return rows

def generate_markdown_report(command: str, baseline_runs: List[Dict], optimized_runs: List[Dict],
                           metrics: Dict, beam_value: str) -> str:
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
        f"\n## 2. Optimized Compute Kernels - BEAM {beam_value}\n",
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

    efficiency = (metrics['total_gflops_optimized'] / metrics['total_gflops_baseline'] - 1) * 100
    time_reduction_ms = metrics['time_reduction'] / 1000.0

    baseline_compute, baseline_transfer, baseline_total = get_total_runtime_with_transfers(best_baseline)
    optimized_compute, optimized_transfer, optimized_total = get_total_runtime_with_transfers(best_optimized)

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

    report.extend(["\n## 5. Kernel Source Code\n"])

    def format_kernel_code(kernel_data: Dict) -> List[str]:
        kernel_sections = []
        for kernel in kernel_data.get("kernels", []):
            if "copy" in kernel.get('kernel_name', '').lower():
                continue

            kernel_name = kernel.get('kernel_name', '')
            code = kernel.get('src')  # Changed from 'code' to 'src'
            if code:
                kernel_sections.extend([
                    f"\n### Kernel: {kernel_name}\n",
                    "```c",
                    code,
                    "```\n"
                ])
        return kernel_sections

    report.extend(format_kernel_code(best_baseline))

    optimized_kernels = {k.get('kernel_name'): k for k in best_optimized.get("kernels", [])
                        if "copy" not in k.get('kernel_name', '').lower()}
    baseline_kernels = {k.get('kernel_name'): k for k in best_baseline.get("kernels", [])
                       if "copy" not in k.get('kernel_name', '').lower()}

    if optimized_kernels != baseline_kernels:
        report.extend(["\n### Optimized Kernels"])
        report.extend(format_kernel_code(best_optimized))

    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Run optimization analysis')
    parser.add_argument('--output-dir', default=None, help='Output directory for analysis files')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs for each phase')
    parser.add_argument('--beam', default=os.environ.get("BEAM", "3"), help='BEAM value for optimization')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or f"performance_reports_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    for command in PROGRAMS:
        print(f"\nAnalyzing {command}...")

        baseline_runs = []
        print("\nPhase 1: Running baseline tests...")
        for i in range(args.num_runs):
            print(f"Baseline run {i+1}/{args.num_runs}")
            baseline_result = run_tinygrad_program(command, optimization=False, output_dir=output_dir, run_index=i, beam_value=args.beam)
            baseline_runs.append(baseline_result)

        print("\nPhase 2: Running optimization step...")
        run_tinygrad_program(command, optimization=True, output_dir=output_dir, run_index=0, beam_value=args.beam)

        print("\nPhase 3: Running optimized tests...")
        for i in range(args.num_runs):
            print(f"Optimized run {i+1}/{args.num_runs}")
            optimized_result = run_tinygrad_program(command, optimization=True, output_dir=output_dir, run_index=i, beam_value=args.beam)
            optimized_runs.append(optimized_result)

        metrics = calculate_performance_metrics(baseline_runs, optimized_runs)

        report = generate_markdown_report(command, baseline_runs, optimized_runs, metrics, args.beam)

        example_name = os.path.basename(command.split()[1]).replace(".py", "")
        report_file = os.path.join(output_dir, f"{example_name}_analysis.md")

        with open(report_file, "w") as f:
            f.write(report)

        print(f"\nReport saved to {report_file}")
        print(f"All files saved in: {output_dir}")

if __name__ == "__main__":
    main()