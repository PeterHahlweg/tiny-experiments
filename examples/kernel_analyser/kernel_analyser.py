from __future__ import annotations
import re, json
from dataclasses import dataclass, asdict
from typing import Optional, List
from tinygrad.helpers import GlobalCounters
from tinygrad.engine.realize import ExecItem

@dataclass
class Metrics:
    memory_GB: float
    timing_us: float
    gflops: Optional[float] = None
    bandwidth: Optional[str] = None
    buffer_count: Optional[int] = None
    lds_bandwidth_GB: Optional[float] = None  # Local Data Share bandwidth

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
        self.hook()

    def hook(self):
        print("Installing hook...")
        self._original_run = ExecItem.run
        def _wrapped_run(exec_item, var_vals=None, *, wait=False, jit=False, do_update_stats=True):
            try:
                et = self._original_run(exec_item, var_vals, wait=wait, jit=jit, do_update_stats=do_update_stats)
                if et is not None and do_update_stats:
                    metrics = Metrics(
                        memory_GB=GlobalCounters.mem_used/1e9,
                        timing_us=et * 1e6,
                        buffer_count=len(exec_item.bufs)
                    )
                    if hasattr(exec_item.prg, 'estimates'):
                        op_est = exec_item.prg.estimates.ops
                        mem_est = exec_item.prg.estimates.mem
                        lds_est = exec_item.prg.estimates.lds
                        if op_est: metrics.gflops = float(op_est)/(et*1e9)
                        if mem_est: metrics.bandwidth = f"{float(mem_est)/(et*1e9):.2f}"
                        if lds_est: metrics.lds_bandwidth_GB = float(lds_est)/(et*1e9)

                    name = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '',
                                exec_item.prg.display_name if hasattr(exec_item.prg, 'display_name') else 'unknown')
                    if 'copy' in name.lower():
                        src_device = exec_item.bufs[0].device
                        dst_device = exec_item.bufs[1].device
                        if src_device == "CLANG": src_device = "NPY"
                        transfer = f"{dst_device} <- {src_device}" if src_device != dst_device else ""
                        name = f"copy {exec_item.bufs[0].size}, {transfer}"

                    kernel = Kernel(
                        backend="METAL",
                        kernel_index=GlobalCounters.kernel_count,
                        kernel_name=name,
                        metrics=metrics,
                        code=exec_item.prg.code if hasattr(exec_item.prg, 'code') else None
                    )
                    self.kernels.append(kernel)
                return et
            except Exception as e:
                print(f"Error in kernel execution: {str(e)}")
                raise
        ExecItem.run = _wrapped_run

    def __del__(self):
        if self._original_run:
            print("Removing hook...")
            ExecItem.run = self._original_run

    def print_summary(self):
        if not self.kernels:
            print("\nNo kernels recorded")
            return
        total_time_us = sum(k.metrics.timing_us for k in self.kernels)
        print(f"\nTotal Runtime: {total_time_us:.2f} us")
        print("\nKernel Details:")
        for k in self.kernels:
            print(f"{k.kernel_name}: {k.metrics.timing_us:.2f}us, {k.metrics.gflops or 0:.2f} GFLOPS")

    def write_json(self, filename: str):
        output = {
            'total_runtime_us': sum(k.metrics.timing_us for k in self.kernels),
            'kernels': [asdict(k) for k in self.kernels]
        }
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)