from __future__ import annotations
import re, json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from tinygrad.helpers import GlobalCounters
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.codegen.kernel import Kernel as TinyKernel
from tinygrad.renderer import ProgramSpec

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
    device: Optional[str] = None
    function_name: Optional[str] = None
    src: Optional[str] = None
    uops: Optional[List[Any]] = None

class KernelAnalyzer:
    def __init__(self):
        self.kernels = []
        self._original_run = None
        self._original_to_program = None
        self.program_infos = {}
        self._hook()

    def _hook(self):
        self._original_run = ExecItem.run
        self._original_to_program = TinyKernel.to_program

        def _wrapped_run(exec_item, var_vals=None, *, wait=False, jit=False, do_update_stats=True):
            try:
                et = self._original_run(exec_item, var_vals, wait=wait, jit=jit, do_update_stats=do_update_stats)
                if et is not None and do_update_stats:
                    self.kernels.append(self._create_kernel(exec_item, self._gather_metrics(exec_item, et)))
                return et
            except Exception as e: raise

        def _wrapped_to_program(tiny_kernel, name_override:Optional[str]=None):
            program = self._original_to_program(tiny_kernel, name_override)
            kernel_name = name_override or program.function_name
            self.program_infos[kernel_name] = {
                'device': program.device,
                'function_name': program.function_name,
                'src': program.src,
                'uops': program.uops
            }
            return program

        ExecItem.run = _wrapped_run
        TinyKernel.to_program = _wrapped_to_program

    def _gather_metrics(self, exec_item, execution_time: float) -> Metrics:
        metrics = Metrics(
            memory_GB=GlobalCounters.mem_used/1e9,
            timing_us=execution_time * 1e6,
            buffer_count=len(exec_item.bufs)
        )
        if hasattr(exec_item.prg, 'estimates'):
            op_est = exec_item.prg.estimates.ops
            mem_est = exec_item.prg.estimates.mem
            lds_est = exec_item.prg.estimates.lds
            if op_est: metrics.gflops = float(op_est)/(execution_time*1e9)
            if mem_est: metrics.bandwidth = f"{float(mem_est)/(execution_time*1e9):.2f}"
            if lds_est: metrics.lds_bandwidth_GB = float(lds_est)/(execution_time*1e9)

        return metrics

    def _create_kernel(self, exec_item, metrics: Metrics) -> Kernel:
        name = self._get_kernel_name(exec_item)
        code = None
        if isinstance(exec_item.prg, CompiledRunner):
            try: code = exec_item.prg._prg.fxn.__str__()
            except Exception: pass
        return Kernel(
            backend="METAL",
            kernel_index=GlobalCounters.kernel_count,
            kernel_name=name,
            metrics=metrics,
            **self.program_infos.get(name, {})
        )

    def _get_kernel_name(self, exec_item) -> str:
        name = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '',
                     exec_item.prg.display_name if hasattr(exec_item.prg, 'display_name') else 'unknown')
        if 'copy' in name.lower():
            src_device = exec_item.bufs[0].device
            dst_device = exec_item.bufs[1].device
            src_device = "NPY" if src_device == "CLANG" else src_device
            transfer = f"{dst_device} <- {src_device}" if src_device != dst_device else ""
            name = f"copy {exec_item.bufs[0].size}, {transfer}"
        return name

    def __del__(self):
        if self._original_run:
            ExecItem.run = self._original_run

    def write_json(self, filename: str):
        output = {
            'total_runtime_us': sum(k.metrics.timing_us for k in self.kernels),
            'kernels': [{
                'backend': k.backend,
                'kernel_index': k.kernel_index,
                'kernel_name': k.kernel_name,
                'device': k.device,
                'src': k.src,
                'metrics': asdict(k.metrics),
            } for k in self.kernels]
        }
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)