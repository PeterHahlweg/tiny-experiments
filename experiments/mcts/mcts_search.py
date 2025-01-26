from __future__ import annotations
from typing import List, Optional, Dict, cast
import numpy as np
np.set_printoptions(suppress=True)
import math, functools, time, random, statistics, json, os
from tinygrad.helpers import DEBUG, getenv, CACHELEVEL, diskcache_get, diskcache_put, colored
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.device import Buffer, Device, CompileError
from tinygrad.engine.search import _ensure_buffer_alloc, get_kernel_actions, _time_program

MULTI_PASS = getenv("MP", 0)
KERNEL_CACHE_FILE = os.path.expanduser("~/.tinygrad/kernel_cache.json")
MIN_IMPROVEMENT = getenv("MIN_IMPROVEMENT", 0.01)

class KernelCache:
    @staticmethod
    def load() -> Dict:
        try:
            if os.path.exists(KERNEL_CACHE_FILE):
                with open(KERNEL_CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                    for v in cache.values():
                        if not isinstance(v.get("pass", 1), int):
                            v["pass"] = 1
                    return cache
        except Exception as e:
            if DEBUG >= 2:
                print(f"Error loading kernel cache: {str(e)}")
        return {}

    @staticmethod
    def save(cache: Dict, kernel_hash: str, kernel: Kernel, new_time: float,
            old_time: float, current_pass: int) -> bool:
        if new_time >= old_time * (1 - MIN_IMPROVEMENT):
            return False

        improvement = (old_time - new_time) / old_time * 100 if old_time != math.inf else 100

        cache[kernel_hash] = {
            "opts": [{"op": opt.op.name, "axis": opt.axis, "amt": opt.amt}
                    for opt in kernel.applied_opts],
            "time": new_time,
            "pass": current_pass + 1,
            "last_improved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "improvement": f"{improvement:.1f}%"
        }

        try:
            os.makedirs(os.path.dirname(KERNEL_CACHE_FILE), exist_ok=True)

            if os.path.exists(KERNEL_CACHE_FILE):
                backup_file = f"{KERNEL_CACHE_FILE}.backup"
                os.replace(KERNEL_CACHE_FILE, backup_file)

            temp_file = f"{KERNEL_CACHE_FILE}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(cache, f)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_file, KERNEL_CACHE_FILE)

            if os.path.exists(f"{KERNEL_CACHE_FILE}.backup"):
                os.remove(f"{KERNEL_CACHE_FILE}.backup")

            return True
        except Exception as e:
            if DEBUG >= 2:
                print(f"Error saving kernel cache: {str(e)}")
            backup_file = f"{KERNEL_CACHE_FILE}.backup"
            if os.path.exists(backup_file):
                os.replace(backup_file, KERNEL_CACHE_FILE)
            return False

class MCTSNode:
    def __init__(self, kernel: Kernel, parent=None):
        self.kernel = kernel
        self.t = math.inf
        self.n = 0
        self.tm = math.inf
        self.i = -1
        self.parents = [parent] if parent else []
        self.children = None
        self.removed_children = []

def expand_node(node: MCTSNode):
    assert node.children is None
    node.children = [MCTSNode(x, node) for x in get_kernel_actions(node.kernel, include_0=False).values()]

def remove_node(node: MCTSNode):
    for parent in node.parents:
        assert parent.children is not None
        parent.children.remove(node)
        parent.removed_children.append(node)

C = math.sqrt(2)
TEMP = 0.5

def _sample_tree(node: MCTSNode, best_tm: float) -> MCTSNode:
    if node.children is None or len(node.children) == 0:
        return node

    unexplored = []
    explored = []
    ucb_vals = []

    for child in node.children:
        if child.n == 0:
            unexplored.append(child)
        else:
            ucb = -child.t/best_tm + C*math.sqrt(math.log(node.n)/child.n)
            explored.append(child)
            ucb_vals.append(ucb)

    if unexplored:
        return random.choice(unexplored)
    if not explored:
        return node

    ucb_exp = np.exp((np.array(ucb_vals)-max(ucb_vals))/TEMP)
    probs = ucb_exp/np.sum(ucb_exp)
    selected = explored[np.random.choice(len(ucb_exp), p=probs)]
    return _sample_tree(selected, best_tm)

def sample_tree(root: MCTSNode, best_tm: float) -> Optional[MCTSNode]:
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

def backprop(node: MCTSNode, tm: float, strength=1.0):
    if node.t > tm:
        node.t = tm
    node.n += strength
    for parent in node.parents:
        backprop(parent, tm, strength/len(node.parents))

def mcts_search(lin: Kernel, rawbufs: List[Buffer], amt: int) -> Kernel:
    # Check regular MCTS cache first
    key = {"ast": lin.ast.key, "amt": amt, "device": lin.opts.device, "suffix": lin.opts.suffix}
    if not getenv("IGNORE_MCTS_CACHE") and CACHELEVEL >= 1 and (val := diskcache_get("mcts_search", key)) is not None:
        ret = lin.copy()
        for o in val[len(lin.applied_opts):]:
            ret.apply_opt(o)
        return ret

    # Initialize search state
    best_tm = math.inf
    cached_time = math.inf
    cached_pass = 0
    start_kernel = lin

    # Load multi-pass cache if enabled
    if MULTI_PASS:
        cache = KernelCache.load()
        kernel_hash = f"{lin.ast.key}_{lin.name}"

        if kernel_hash in cache:
            cached = cache[kernel_hash]
            start_kernel = lin.copy()
            for opt_dict in cached["opts"]:
                opt = Opt(OptOps[opt_dict["op"]], opt_dict["axis"], opt_dict["amt"])
                start_kernel.apply_opt(opt)
            cached_time = cached["time"]
            cached_pass = cached.get("pass", 1)
            best_tm = cached_time  # Start with cached time as best

    # Prepare for search
    rawbufs = _ensure_buffer_alloc(rawbufs)
    var_vals = {k: (k.vmax + k.vmin)//2 for k in lin.ast.variables()}
    dev = Device[lin.opts.device]
    root = MCTSNode(start_kernel)

    st = time.perf_counter()
    best, best_idx = start_kernel, 0
    seen_libs: Dict[bytes, MCTSNode] = {}
    seen_asts: Dict[bytes, MCTSNode] = {}
    compile_time = runtime_time = 0.0

    for i in range(amt):
        node = sample_tree(root, best_tm)
        if node is None:
            break
        node.i = i

        opt_ast = node.kernel.get_optimized_ast()
        if (sibling := seen_asts.get(opt_ast.key)):
            remove_node(node)
            tm = sibling.t
        else:
            seen_asts[opt_ast.key] = node
            p = node.kernel.to_program(name_override="test")

            tm1 = time.perf_counter()
            try:
                lib = dev.compiler.compile(p.src)
            except CompileError:
                lib = None
            tm2 = time.perf_counter()

            if lib is None:
                tm = math.inf
            else:
                if (sibling := seen_libs.get(lib)):
                    remove_node(node)
                    tm = sibling.t
                else:
                    seen_libs[lib] = node
                    try:
                        tm = statistics.median(_time_program(p, lib, var_vals, rawbufs, cnt=3, early_stop=best_tm*5/1e6))*1e6
                        if tm < 0:  # Handle negative times
                            tm = best_tm * 10  # 10x worse than current best
                    except RuntimeError:
                        tm = best_tm * 10
                    node.tm = tm
            tm3 = time.perf_counter()
            compile_time += tm2-tm1
            runtime_time += tm3-tm2

        if tm < best_tm:
            best, best_idx, best_tm = node.kernel, i, tm
            # Save immediately when we find a better kernel
            if MULTI_PASS and tm < cached_time:
                KernelCache.save(cache, kernel_hash, node.kernel, tm, cached_time, cached_pass)

        if DEBUG >= 2:
            et = time.perf_counter() - st
            print(f"\r{et:7.2f}s {colored(f'{compile_time*100/et:3.0f}%', 'cyan')} "
                  f"{colored(f'{runtime_time*100/et:3.0f}%', 'red')}: {tm:12.2f} us "
                  f"best: {best_tm:12.2f} us @ {best_idx+1:4d} {i+1:4d}/{amt:4d} "
                  f"{int(round((i+1)/et)):4d}/s {node.kernel.colored_shape()}"
                  f"{' [MP'+str(cached_pass+1)+']' if MULTI_PASS else ''}\033[K", end="")

        backprop(node, tm)

    if DEBUG >= 2:
        print()

    if CACHELEVEL >= 1:
        diskcache_put("mcts_search", key, best.applied_opts)
    return best