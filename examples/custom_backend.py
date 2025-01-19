from __future__ import annotations
import numpy as np
from tinygrad.ops import Ops
from tinygrad.device import Compiled, Allocator, BufferSpec, Device

# NOTE: This is a simplified version that piggybacks on the existing PYTHON device.
# For a proper custom backend, you would need to implement ops_<device>.py in tinygrad/runtime/
# See tinygrad/runtime/ops_cpu.py for an example

class NumpyAllocator(Allocator):
  def _alloc(self, size:int, options:BufferSpec) -> Any: return np.zeros(size, dtype=np.float32)
  def _copyin(self, dest, src:memoryview): np.copyto(dest, np.frombuffer(src, dtype=np.float32))
  def _copyout(self, dest:memoryview, src): np.copyto(np.frombuffer(dest, dtype=np.float32), src)

class NumpyDevice(Compiled):
  def __init__(self): super().__init__("PYTHON", NumpyAllocator(), None, None, None)
  def exec_ast(self, ast, buf):
    def get_buffer(x): return x._buf if hasattr(x, '_buf') else x

    if ast.op == Ops.NOOP: return get_buffer(ast.arg[0])
    if ast.op == Ops.COPY:
      if buf is not None:
        np.copyto(get_buffer(buf), get_buffer(ast.arg[0]))
        return buf
      return get_buffer(ast.arg[0]).copy()

    inputs = [get_buffer(x) for x in ast.arg]
    ops = {
      Ops.ADD: np.add,
      Ops.SUB: np.subtract,
      Ops.MUL: np.multiply,
      Ops.DIV: np.divide,
      Ops.SUM: lambda x,*_: np.sum(x),
      Ops.MAX: lambda x,*_: np.max(x),
      Ops.EXP2: np.exp2,
      Ops.LOG2: np.log2,
      Ops.SQRT: np.sqrt,
    }

    if ast.op not in ops: raise NotImplementedError(f"op {ast.op} not implemented")
    ret = ops[ast.op](*inputs)

    if buf is not None:
      np.copyto(get_buffer(buf), ret)
      return buf
    return ret

if __name__ == "__main__":
  from tinygrad import Tensor

  # Create test tensors
  a = Tensor([1,2,3,4], device="PYTHON")
  b = Tensor([4,3,2,1], device="PYTHON")

  print("\nTesting basic arithmetic...")
  print("ADD:", (a + b).numpy())        # [5,5,5,5]
  print("SUB:", (a - b).numpy())        # [-3,-1,1,3]
  print("MUL:", (a * b).numpy())        # [4,6,6,4]
  print("DIV:", (a / b).numpy())        # [0.25,0.67,1.5,4]

  print("\nTesting reductions...")
  print("SUM:", a.sum().numpy())        # 10
  print("MAX:", a.max().numpy())        # 4

  print("\nTesting unary operations...")
  print("EXP2:", a.exp2().numpy())      # [2,4,8,16]
  print("LOG2:", a.log2().numpy())      # [0,1,1.58,2]
  print("SQRT:", a.sqrt().numpy())      # [1,1.41,1.73,2]