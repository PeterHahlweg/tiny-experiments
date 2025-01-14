#!/usr/bin/env python
import re
import sys
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

@dataclass
class Metrics:
    memory_GB: float
    timing_us: float
    gflops: Optional[float] = None
    bandwidth: Optional[str] = None

@dataclass
class Kernel:
    backend: str
    kernel_index: int
    kernel_name: str
    metrics: Metrics
    code: Optional[str] = None
    pre_kernel_output: Optional[str] = None

def clean_ansi(text: str, output_file: Optional[Path] = None) -> str:
    """Remove ANSI escape codes from text."""
    cleaned = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
    if output_file:
        output_file.write_text(cleaned)
    return cleaned

def parse_metrics(parts: list[str]) -> Metrics:
    """Extract performance metrics from kernel info line parts."""
    metrics = {}
    try:
        metrics['memory_GB'] = float(parts[parts.index('mem') + 1])
        
        timing = parts[parts.index('tm') + 1].rstrip('/') 
        metrics['timing_us'] = float(timing[:-2]) * (1000 if timing.endswith('ms') else 1)
        
        if 'GFLOPS' in parts:
            metrics['gflops'] = float(parts[parts.index('GFLOPS') - 1])
        if 'GB/s' in parts:
            metrics['bandwidth'] = parts[parts.index('GB/s') - 1]
            
    except (ValueError, IndexError):
        pass
    
    return Metrics(**metrics)

def parse_kernel(line: str, pre_lines: list[str]) -> Kernel:
    """Parse a kernel info line into a Kernel object."""
    parts = line.replace('***', '').strip().split()
    
    # Extract kernel name
    name_parts = []
    for part in parts[2:]:
        if part == 'arg': break
        name_parts.append(part)
        
    kernel = Kernel(
        backend=parts[0],
        kernel_index=int(parts[1]),
        kernel_name=' '.join(name_parts).strip(),
        metrics=parse_metrics(parts)
    )
    
    if pre_lines:
        text = '\n'.join(pre_lines)
        if '#include' in text:
            kernel.pre_kernel_output = text.split('#include')[0].strip()
            kernel.code = '#include' + text.split('#include', 1)[1]
            
    return kernel

def process_log(content: str) -> list[Kernel]:
    """Process log content into list of Kernel objects."""
    kernels = []
    pre_lines = []
    
    for line in content.splitlines():
        if not line: continue
            
        if '***' in line and 'GFLOPS' in line and 'GB/s' in line:
            kernels.append(parse_kernel(line, pre_lines))
            pre_lines = []
        else:
            pre_lines.append(line)
            
    return kernels

def main():
    if len(sys.argv) < 2:
        print("Usage: analyse.py LOG_FILE [OUTPUT_FILE]")
        sys.exit(1)
        
    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    try:
        content = clean_ansi(log_path.read_text(), log_path.with_suffix('.cleaned'))
        kernels = process_log(content)
        total_runtime = sum(k.metrics.timing_us for k in kernels)
        output = json.dumps({
            "total_runtime_us": total_runtime,
            "kernels": [asdict(k) for k in kernels]
        }, indent=2)
        
        if out_path:
            out_path.write_text(output)
            print(f"JSON output written to {out_path}")
        else:
            print(output)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
