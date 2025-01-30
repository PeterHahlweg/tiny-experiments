from __future__ import annotations
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass
import numpy as np
import math
import time
import random
from tinygrad.helpers import DEBUG, getenv, CACHELEVEL, diskcache_get, diskcache_put, colored
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import Buffer, Device, CompileError
from tinygrad.engine.search import _ensure_buffer_alloc, get_kernel_actions, _time_program

# Configuration
REGION_SEARCH = getenv("RS", 0)  # Enable Region Search: RS=0 disabled, RS=1 depth 1, RS=2 depth 2, etc
MAX_TIMEOUTS = getenv("MAX_TIMEOUTS", 3)  # Maximum allowed timeouts before stopping
MAX_TIME_MS = getenv("MAX_TIME_MS", 10)  # Maximum allowed kernel execution time in ms
# RS value directly maps to search depth
MAX_SEARCH_DEPTH = REGION_SEARCH if REGION_SEARCH > 0 else 0

@dataclass
class SearchNode:
    kernel: Kernel
    time: float
    depth: int = 0

class MCTSNode:
    def __init__(self, kernel: Kernel, parent=None):
        self.kernel: Kernel = kernel
        self.t = math.inf  # Best time found
        self.n = 0  # Number of visits
        self.tm = math.inf  # Time for this node
        self.i = -1  # When was this node explored
        self.parents: List[MCTSNode] = [parent] if parent is not None else []
        self.children: Optional[List[MCTSNode]] = None
        self.removed_children: List[MCTSNode] = []

def safe_execute(program, lib, var_vals, rawbufs, best_time: float) -> float:
    """Execute kernel multiple times to get reliable timing"""
    try:
        # Keep reasonable timeout threshold
        early_stop = min(best_time * 2, MAX_TIME_MS * 1000) / 1e6  # Convert to seconds

        # Run multiple times and take minimum
        times = _time_program(
            program, lib, var_vals, rawbufs,
            cnt=3,  # Run 3 times
            early_stop=early_stop
        )
        return min(times) * 1e6

    except Exception as e:
        if DEBUG >= 3:
            print(f"\rExecution error: {str(e)}")
        return float('inf')

def build_search_tree(kernel: Kernel, max_depth: int) -> List[Tuple[Kernel, int, List[str]]]:
    """Build complete tree of kernels to explore up to max_depth, with applied optimizations"""
    all_kernels = []
    seen_asts = {kernel.get_optimized_ast().key}  # Start with the MCTS best kernel in seen set
    kernels_by_depth = {1: []}  # Start from depth 1

    # First get all direct neighbors of starting kernel
    actions = get_kernel_actions(kernel, include_0=False)
    for opt_name, new_kernel in actions.items():
        opt_ast = new_kernel.get_optimized_ast()
        if opt_ast.key not in seen_asts:
            seen_asts.add(opt_ast.key)
            kernels_by_depth[1].append((new_kernel, [opt_name]))

    # Then discover subsequent depths
    for depth in range(1, max_depth):
        kernels_by_depth[depth + 1] = []

        # Process all kernels at current depth
        for current_kernel, current_opts in kernels_by_depth[depth]:
            actions = get_kernel_actions(current_kernel, include_0=False)

            # Get all possible next optimizations
            for opt_name, new_kernel in actions.items():
                opt_ast = new_kernel.get_optimized_ast()
                if opt_ast.key not in seen_asts:
                    seen_asts.add(opt_ast.key)
                    new_opts = current_opts + [opt_name]
                    kernels_by_depth[depth + 1].append((new_kernel, new_opts))

    # Now flatten into execution order, starting from depth 1
    for depth in range(1, max_depth + 1):
        for kernel, opts in kernels_by_depth[depth]:
            all_kernels.append((kernel, depth, opts))

    return all_kernels

def deep_region_search(best_kernel: Kernel, best_time: float, dev: Device,
                      rawbufs: List[Buffer], var_vals: dict,
                      seen_asts: Set[bytes]) -> Tuple[Kernel, float]:
    """Enhanced region search that explores neighbors up to configurable depth"""
    # First build the complete tree of kernels to explore
    all_kernels = build_search_tree(best_kernel, MAX_SEARCH_DEPTH)
    total_kernels = len(all_kernels)
    current_best = (best_kernel, best_time, 0)  # Added index to track where best was found
    timeout_count = 0
    explored = 0
    compile_time = runtime_time = 0.0
    st = time.perf_counter()

    for new_kernel, depth, opts in all_kernels:
        explored += 1

        try:
            tm1 = time.perf_counter()
            program = new_kernel.to_program()
            lib = dev.compiler.compile(program.src)
            tm2 = time.perf_counter()

            tm = safe_execute(program, lib, var_vals, rawbufs, current_best[1])
            tm3 = time.perf_counter()

            compile_time += tm2-tm1
            runtime_time += tm3-tm2

            if tm == float('inf'):
                timeout_count += 1
                if timeout_count >= MAX_TIMEOUTS:
                    if DEBUG >= 3:
                        print(f"\nStopping due to {timeout_count} timeouts")
                    break
                continue

            # Update best if we found better performance
            if tm < current_best[1]:
                current_best = (new_kernel, tm, explored)

            # Print progress line similar to MCTS
            if DEBUG >= 2:
                et = time.perf_counter() - st
                print(f"\rRS   [{et:7.2f}s] {colored(f'{compile_time*100/et:3.0f}%', 'cyan')} "
                      f"{colored(f'{runtime_time*100/et:3.0f}%', 'red')}: {tm:12.2f} us     "
                      f"best: {current_best[1]:12.2f} us @ {current_best[2]+1:4d}      "
                      f"{explored:4d}/{total_kernels:4d}  {int(round(explored/et)):4d}/s     "
                      f"{new_kernel.colored_shape()}\033[K", end="")

        except CompileError:
            continue

    if DEBUG >= 2:
        print()

    return current_best[0], current_best[1]

def expand_node(node: MCTSNode):
    """Expand a node by creating children for all possible kernel actions"""
    assert node.children is None
    node.children = [MCTSNode(x, node) for x in get_kernel_actions(node.kernel, include_0=False).values()]

def remove_node(node: MCTSNode):
    """Remove a node from its parents"""
    for parent in node.parents:
        assert parent.children is not None
        parent.children.remove(node)
        parent.removed_children.append(node)

# MCTS constants
C = math.sqrt(2)  # Exploration constant
TEMP = 0.5  # Temperature for softmax

def _sample_tree(node: MCTSNode, best_tm: float) -> MCTSNode:
    """Sample a node from the tree using UCB1"""
    if node.children is None or len(node.children) == 0:
        return node

    unexplored_children = []
    explored_children = []
    ucb_explored_children: List[float] = []

    for child in node.children:
        if child.n == 0:
            unexplored_children.append(child)
        else:
            ucb = -child.t/best_tm + C*math.sqrt(math.log(node.n)/child.n)
            if not math.isinf(ucb):
                explored_children.append(child)
                ucb_explored_children.append(ucb)

    if len(unexplored_children):
        return random.choice(unexplored_children)
    if not len(explored_children):
        return node

    # Softmax selection
    ucb_exp = np.exp((np.array(ucb_explored_children)-max(ucb_explored_children))/TEMP)
    return _sample_tree(
        explored_children[np.random.choice(len(ucb_exp), p=ucb_exp/np.sum(ucb_exp))],
        best_tm
    )

def sample_tree(root: MCTSNode, best_tm: float) -> Optional[MCTSNode]:
    """Sample and expand the tree"""
    if root.children is None:
        expand_node(root)
    while root.children:
        node = _sample_tree(root, best_tm)

        if node.children is not None and len(node.children) == 0:
            remove_node(node)
            continue

        if node.n != 0:
            if node.children is None:
                expand_node(node)
            assert node.children is not None
            if len(node.children) == 0:
                remove_node(node)
                continue
            node = random.choice(node.children)
        return node
    return None

def backprop(node: MCTSNode, tm: float, strength: float = 1.0):
    """Backpropagate results through the tree"""
    if node.t > tm:
        node.t = tm
    node.n += strength
    for parent in node.parents:
        backprop(parent, tm, strength/len(node.parents))

def run_mcts(lin: Kernel, rawbufs: List[Buffer], amt: int, dev: Device,
             var_vals: dict) -> Tuple[MCTSNode, Kernel, float]:
    """Run MCTS search to find the best kernel"""
    root = MCTSNode(lin)
    best, best_idx, best_tm = lin, 0, math.inf
    seen_libs: Dict[bytes, MCTSNode] = {}
    seen_asts: Dict[bytes, MCTSNode] = {}
    compile_time = runtime_time = 0.0
    st = time.perf_counter()

    for i in range(amt):
        node = sample_tree(root, best_tm)
        if node is None:
            break
        node.i = i

        opt_ast = node.kernel.get_optimized_ast()
        if (sibling_node := seen_asts.get(opt_ast.key, None)) is not None:
            remove_node(node)
            tm = sibling_node.t
        else:
            seen_asts[opt_ast.key] = node
            program = node.kernel.to_program()

            tm1 = time.perf_counter()
            try:
                lib = dev.compiler.compile(program.src)
            except CompileError:
                lib = None
            tm2 = time.perf_counter()

            if lib is None:
                tm = math.inf
            else:
                if (sibling_node := seen_libs.get(lib, None)) is not None:
                    remove_node(node)
                    tm = sibling_node.t
                else:
                    seen_libs[lib] = node
                    try:
                        tm = safe_execute(program, lib, var_vals, rawbufs, best_tm)
                    except Exception:
                        tm = math.inf
                    node.tm = tm

            tm3 = time.perf_counter()
            compile_time += tm2-tm1
            runtime_time += tm3-tm2

        if tm < best_tm:
            best, best_idx, best_tm = node.kernel, i, tm

        et = time.perf_counter() - st
        if DEBUG >= 2:
            print(f"\rMCTS [{et:7.2f}s] {colored(f'{compile_time*100/et:3.0f}%', 'cyan')} "
                  f"{colored(f'{runtime_time*100/et:3.0f}%', 'red')}: {tm:12.2f} us     "
                  f"best: {best_tm:12.2f} us @ {best_idx+1:4d}      "
                  f"{i+1:4d}/{amt:4d}  {int(round((i+1)/et)):4d}/s     "
                  f"{node.kernel.colored_shape()}\033[K", end="")

        backprop(node, tm)

    if DEBUG >= 2:
        print()

    return root, best, best_tm

def mcts_search(lin: Kernel, rawbufs: List[Buffer], amt: int) -> Kernel:
    """Two-stage kernel optimization with deep region search"""
    # Check cache first
    key = {
        "ast": lin.ast.key,
        "amt": amt,
        "device": lin.opts.device,
        "suffix": lin.opts.suffix
    }

    if not getenv("IGNORE_MCTS_CACHE") and CACHELEVEL >= 1:
        if (val := diskcache_get("mcts_search", key)) is not None:
            ret = lin.copy()
            for o in val[len(lin.applied_opts):]:
                ret.apply_opt(o)
            return ret

    # Initialize search
    rawbufs = _ensure_buffer_alloc(rawbufs)
    var_vals = {k:(k.vmax+k.vmin)//2 for k in lin.ast.variables()}
    dev = Device[lin.opts.device]

    # Stage 1: Run MCTS search
    root, best_kernel, best_tm = run_mcts(lin, rawbufs, amt, dev, var_vals)

    # Stage 2: Deep region search
    if REGION_SEARCH:
        seen_asts = {best_kernel.get_optimized_ast().key}
        try:
            final_kernel, final_tm = deep_region_search(
                best_kernel, best_tm,
                dev,
                rawbufs,
                var_vals,
                seen_asts
            )

            best_kernel = final_kernel

        except Exception as e:
            if DEBUG >= 2:
                print(f"\nRegion search failed: {str(e)}")

    # Cache the result
    if CACHELEVEL >= 1:
        diskcache_put("mcts_search", key, best_kernel.applied_opts)

    return best_kernel