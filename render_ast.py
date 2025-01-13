from tinygrad.codegen.kernel import Kernel
from tinygrad.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import dtypes
from tinygrad.shape.shapetracker import ShapeTracker, View

# AST from UOps
ast_r_64_32_16_3_4_3_3_3 = UOp(Ops.SINK, dtypes.void, arg=KernelInfo(local_dims=2, upcasted=4, dont_use_locals=False), src=(
 UOp(Ops.STORE, dtypes.void, arg=None, src=(
   UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(518400), arg=0, src=()),
   UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64, 32, 16, 3, 4, 1, 3, 1), strides=(8640, 64, 4, 2880, 1, 0, 960, 0), offset=0, mask=((0, 60), (0, 15), (0, 16), (0, 3), (0, 4), (0, 1), (0, 3), (0, 1)), contiguous=False),)), src=()),
   UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 7)), src=(
     UOp(Ops.MUL, dtypes.float, arg=None, src=(
       UOp(Ops.LOAD, dtypes.float, arg=None, src=(
         UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(518400), arg=1, src=()),
         UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1, 1, 4, 542, 4, 962), strides=(0, 0, 0, 0, 0, 960, 0, 1), offset=-961, mask=((0, 1), (0, 1), (0, 1), (0, 1), (0, 4), (1, 541), (0, 4), (1, 961)), contiguous=False), View(shape=(64, 32, 16, 3, 4, 3, 3, 3), strides=(34632, 64, 4, 11544, 1, 963, 3848, 2089464), offset=0, mask=((0, 60), (0, 15), (0, 16), (0, 3), (0, 4), (0, 3), (0, 3), (0, 3)), contiguous=False))), src=()),)),
       UOp(Ops.LOAD, dtypes.float, arg=None, src=(
         UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(9), arg=2, src=()),
         UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64, 32, 16, 3, 4, 3, 3, 3), strides=(0, 0, 0, 0, 0, 1, 0, 3), offset=0, mask=((0, 60), (0, 15), (0, 16), (0, 3), (0, 4), (0, 3), (0, 3), (0, 3)), contiguous=False),)), src=()),)),)),)),)),))


# Get the unoptimized program
kernel = Kernel(ast_r_64_32_16_3_4_3_3_3)
program = kernel.to_program()
print("Unoptimized Kernel Code for r_64_32_16_3_4_3_3_3:")
print(program.src)
