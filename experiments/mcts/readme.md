# Kernel Search Optimizations

This folder contains two independent features that enhance kernel search optimization: Region Search and Multi-Pass Search.

# Region Search for Kernel Optimization

Region Search (RS) is an optional second stage for kernel optimization that runs after Monte Carlo Tree Search (MCTS). It systematically explores the neighborhood around the best kernel found by MCTS by testing additional optimization combinations up to a configurable depth.

## How it Works

1. MCTS runs first and finds a kernel with good performance
2. If enabled (RS > 0), Region Search then:
   - Takes the best kernel from MCTS as starting point
   - Explores all possible optimization combinations up to the depth specified by RS
   - Maintains a set of seen ASTs to avoid duplicating work
   - Updates the best kernel if better performance is found

## Configuration

Configure via environment variables:
- `RS`: Search depth (0 disables RS, 1 for depth 1, 2 for depth 2, etc.)
- `MAX_TIMEOUTS`: Maximum allowed timeouts (default: 3)
- `MAX_TIME_MS`: Maximum allowed kernel execution time in ms (default: 10)

## Implementation Details

The search is implemented in `deep_region_search()` which:
- Builds a complete tree of kernels to explore using `build_search_tree()`
- Tests each kernel in breadth-first order up to the configured depth
- Uses `safe_execute()` to reliably time kernel performance
- Stops early if too many timeouts occur
- Returns the best kernel and timing found

## Usage

Region Search runs automatically as part of `mcts_search()` when enabled:

```python
# Enable depth 2 region search
os.environ["RS"] = "2"

# Run optimization
best_kernel = mcts_search(kernel, rawbufs, amt)
```

## Performance Considerations

- Region Search adds compile and execution time proportional to the number of kernel variants explored
- The number of variants grows exponentially with search depth
- Early stopping on timeouts helps avoid wasting time on slow kernels
- Results are cached to avoid recomputing for identical kernels

# Multi-Pass Search

Multi-Pass Search enables continuing optimization across multiple runs by maintaining a cache of the best kernels found. Each new search starts from the best previously discovered configuration.

## Configuration

Configure via environment variables:
- `MP`: Enable multi-pass search (0 = disabled, 1 = enabled)
- `MIN_IMPROVEMENT`: Minimum relative improvement required to update cache (default: 0.01)

## Implementation Details

The search tracks progress in `~/.tinygrad/kernel_cache.json` using this format:
```json
{
  "kernel_hash": {
    "opts": [{"op": "op_name", "axis": axis, "amt": amount}, ...],
    "time": execution_time,
    "pass": pass_number,
    "last_improved": "YYYY-MM-DD HH:MM:SS",
    "improvement": "X.X%"
  }
}
```

Key features:
- Atomic cache updates with backup for safety
- Immediate cache updates when finding improvements
- Pass counting to track optimization history
- Improvement tracking relative to previous best

## Usage

Multi-Pass Search runs automatically as part of `mcts_search()` when enabled:

```python
# Enable multi-pass search
os.environ["MP"] = "1"

# Run optimization
best_kernel = mcts_search(kernel, rawbufs, amt)
```