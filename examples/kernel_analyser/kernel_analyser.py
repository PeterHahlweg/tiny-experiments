from __future__ import annotations
import re, json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from tinygrad.helpers import GlobalCounters
from tinygrad.engine.realize import ExecItem, CompiledRunner

@dataclass
class Metrics:
    memory_GB: float
    timing_us: float
    gflops: Optional[float] = None
    bandwidth: Optional[str] = None
    buffer_count: Optional[int] = None
    lds_bandwidth_GB: Optional[float] = None

@dataclass
class Kernel:
    backend: str
    kernel_index: int
    kernel_name: str
    metrics: Metrics
    code: Optional[str] = None
    pre_kernel_output: Optional[str] = None

class KernelAnalyser:
    def __init__(self):
        self.kernels: List[Kernel] = []
        self._original_run = None
        self._hook()

    def _hook(self):
        """Install hook to capture kernel metrics"""
        print("Installing hook...")
        self._original_run = ExecItem.run

        def _wrapped_run(exec_item, var_vals=None, *, wait=False, jit=False, do_update_stats=True):
            try:
                et = self._original_run(exec_item, var_vals, wait=wait, jit=jit, do_update_stats=do_update_stats)
                if et is not None and do_update_stats:
                    metrics = self._gather_metrics(exec_item, et)
                    kernel = self._create_kernel(exec_item, metrics)
                    self.kernels.append(kernel)
                return et
            except Exception as e:
                print(f"Error in kernel execution: {str(e)}")
                raise

        ExecItem.run = _wrapped_run

    def _gather_metrics(self, exec_item, execution_time: float) -> Metrics:
        """Gather metrics for a kernel execution"""
        metrics = Metrics(
            memory_GB=GlobalCounters.mem_used/1e9,
            timing_us=execution_time * 1e6,
            buffer_count=len(exec_item.bufs)
        )

        if hasattr(exec_item.prg, 'estimates'):
            op_est = exec_item.prg.estimates.ops
            mem_est = exec_item.prg.estimates.mem
            lds_est = exec_item.prg.estimates.lds

            if op_est:
                metrics.gflops = float(op_est)/(execution_time*1e9)
            if mem_est:
                metrics.bandwidth = f"{float(mem_est)/(execution_time*1e9):.2f}"
            if lds_est:
                metrics.lds_bandwidth_GB = float(lds_est)/(execution_time*1e9)

        return metrics

    def _create_kernel(self, exec_item, metrics: Metrics) -> Kernel:
        name = self._get_kernel_name(exec_item)
        code = None

        if isinstance(exec_item.prg, CompiledRunner):
            try:
                print(f"\nDebug - fxn.__str__: {exec_item.prg._prg.fxn.__str__()}")
                code = exec_item.prg._prg.fxn.__str__()
            except Exception as e:
                print(f"Error getting code: {e}")

        return Kernel(
            backend="METAL",
            kernel_index=GlobalCounters.kernel_count,
            kernel_name=name,
            metrics=metrics,
            code=code
        )

    def _get_kernel_name(self, exec_item) -> str:
        """Get the kernel name, handling copy operations specially"""
        name = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '',
                     exec_item.prg.display_name if hasattr(exec_item.prg, 'display_name') else 'unknown')

        if 'copy' in name.lower():
            src_device = exec_item.bufs[0].device
            dst_device = exec_item.bufs[1].device
            if src_device == "CLANG":
                src_device = "NPY"
            transfer = f"{dst_device} <- {src_device}" if src_device != dst_device else ""
            name = f"copy {exec_item.bufs[0].size}, {transfer}"

        return name

    def __del__(self):
        """Clean up the hook on deletion"""
        if self._original_run:
            print("Removing hook...")
            ExecItem.run = self._original_run

    def print_summary(self):
        """Print a summary of kernel executions"""
        if not self.kernels:
            print("\nNo kernels recorded")
            return

        total_time_us = sum(k.metrics.timing_us for k in self.kernels)
        print(f"\nTotal Runtime: {total_time_us:.2f} us")
        print("\nKernel Details:")
        for k in self.kernels:
            print(f"{k.kernel_name}: {k.metrics.timing_us:.2f}us, {k.metrics.gflops or 0:.2f} GFLOPS")

    def write_json(self, filename: str):
        """Write kernel analysis results to JSON file"""
        output = {
            'total_runtime_us': sum(k.metrics.timing_us for k in self.kernels),
            'kernels': [asdict(k) for k in self.kernels]
        }
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)