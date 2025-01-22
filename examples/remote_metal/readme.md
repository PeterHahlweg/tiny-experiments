# Remote Metal Runtime
## The Problem
Tinygrad has a problem: the current practices of chip vendors.
They don't just sell the hardware - they also sell you their software. This wouldn't be so bad if they allowed customers to actually program the hardware. Instead, they try to hide the hardware interface behind proprietary software and sell this as an advantage.
It's not. Most of the time, it's not even good software. Yes, they try to protect their IP, but if the hardware interface is well-designed, why even care?
The sad thing is that this is the majority approach, as far as I can tell. Some exceptions may exist.
TLDR: You have to buy a license for an x86 compiler in most cases, and it won't run on an ARM chip to optimize your kernels.
## Solutions
### Create Your Own Compiler
This might be possible, but it requires an enormous amount of work. If you're lucky, your hardware is already supported by LLVM, but it most likely isn't the latest and greatest chip.
### Separate Compilation and Execution
The best way around it is:

1. Writing your own runtime for the hardware you have
2. Calling the proprietary compiler for kernel compilation
3. Sending the kernel over ethernet (or whatever transport you prefer)
4. Receiving it with a small C program that reads the socket and uses the vendor API to execute the kernel

To be clear, this is far from optimal for obvious reasons. But if you have no choice, this is probably better than the suboptimal handcrafted kernels from vendors who aren't willing to tell you anything about the hardware. But guess what? Tinygrad will figure out how to run it faster without additional information... let it cook.
## The Experiment
We extend the Metal runtime (ops_metal.py), using its capability to do the compilation and overwrite the MetalProgram. When a MetalProgram is __call__ed, it usually pushes the kernel into the Metal device command queue. We send the kernel along with the input data via TCP to the runner, a small Python script that uses Tinygrad to execute the kernel.
After execution, the output and kernel runtime are sent back and displayed as always. In the runner, we can see any kernel execution along with its runtime, which is nice.
## Result
You can run your programs as usual and also run them with BEAM or MCTS if you want to optimize the kernels, and it will simply work.
You only need to import the module that will piggyback on the Metal runtime.
It's not perfect and not as fast as it could be - maybe a bit hacky in this case - but it solves the problem.
For the runner, you don't need Tinygrad, only the libs from the vendor (Apple in this case) and a TCP socket.
If you have different hardware, try rewriting it.