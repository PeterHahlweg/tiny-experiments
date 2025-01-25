"""
OpsAnalyzer - Simple operation counter for tinygrad kernels

Counts operation types (ADD, MUL, etc) in tinygrad kernels.
Usage:
    analyzer = OpsAnalyzer()
    # Run your tinygrad code
    analyzer.print_summary()
"""
from collections import Counter
from tinygrad.renderer.cstyle import CUDARenderer, OpenCLRenderer, MetalRenderer, ClangRenderer

class OpsAnalyzer:
    def __init__(self):
        self.total_ops = Counter()
        self._original_renders = {}
        for r in [CUDARenderer, OpenCLRenderer, MetalRenderer, ClangRenderer]:
            if hasattr(r, 'render'):
                self._original_renders[r] = r.render
                r.analyzer = self
                orig_render = r.render
                r.render = lambda self, name, uops: (self.analyzer.analyze_uops(uops), orig_render(self, name, uops))[1]

    def __del__(self):
        for r, orig in self._original_renders.items():
            r.render = orig
            delattr(r, 'analyzer')

    def analyze_uops(self, uops):
        self.total_ops.update(uop.op.name for uop in uops)

    def print_summary(self):
        total = sum(self.total_ops.values())
        print("\nOperation Counts (Sorted by Frequency):")
        print(f"{'Operation':<20} {'Count':>8} {'Percentage':>10}")
        print("-" * 40)
        for op, count in self.total_ops.most_common():
            print(f"{op:<20} {count:>8} {count/total:>9.2%}")