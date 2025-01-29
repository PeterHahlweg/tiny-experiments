from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Any
from tinygrad.device import LRUAllocator, BufferSpec, Device, Buffer
import time

@dataclass
class AllocationStats:
    size: int
    creation_time: float
    last_used: float
    hits: int = 0
    misses: int = 0
    allocs: int = 0
    frees: int = 0
    reuses: int = 0

    def to_dict(self):
        return {
            "size_bytes": self.size,
            "size_kb": self.size / 1024,
            "creation_time": self.creation_time,
            "last_used": self.last_used,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "allocations": self.allocs,
            "frees": self.frees,
            "cache_reuses": self.reuses,
            "reuse_rate": (self.reuses / self.allocs * 100) if self.allocs else 0
        }

class MemoryAnalyzer:
    def __init__(self, debug=False):
        self.debug = debug
        self.stats = {}
        self.total = 0
        self.peak = 0
        self.current = 0
        self._wrapped = set()

        # Store original methods
        self._orig_getitem = Device.__getitem__
        self._orig_init = Buffer.__init__
        self._orig_allocs = {}

        # Wrap Buffer init
        def wrap_init(buf_self, device: str, size: int, dtype, *args, **kwargs):
            if self.debug:
                print(f"Buffer init: {device=}, {size=}, {dtype=}")
            self._orig_init(buf_self, device, size, dtype, *args, **kwargs)
            self._track_alloc(device, size * dtype.itemsize, kwargs.get('options'))

        Buffer.__init__ = wrap_init

        # Wrap Device getitem
        def wrap_getitem(dev_self, ix: str):
            device = self._orig_getitem(dev_self, ix)
            if ix not in self._wrapped:
                if self.debug:
                    print(f"Wrapping device: {ix}")

                orig_alloc = device.allocator.alloc
                orig_free = device.allocator.free

                def wrap_alloc(size: int, options: Optional[BufferSpec] = None):
                    actual_size = size
                    if options and hasattr(options, 'dtype'):
                        actual_size *= options.dtype.itemsize

                    if self.debug:
                        print(f"Alloc on {ix}: {actual_size} bytes")

                    cache_key = (size, options)
                    pre_cache = device.allocator.cache.get(cache_key, [])
                    result = orig_alloc(size, options)
                    post_cache = device.allocator.cache.get(cache_key, [])

                    if len(post_cache) < len(pre_cache):
                        self._track_hit(ix, actual_size, options)
                    else:
                        self._track_miss(ix, actual_size, options)

                    self._track_alloc(ix, actual_size, options)
                    return result

                def wrap_free(opaque: Any, size: int, options: Optional[BufferSpec] = None):
                    actual_size = size
                    if options and hasattr(options, 'dtype'):
                        actual_size *= options.dtype.itemsize

                    if self.debug:
                        print(f"Free on {ix}: {actual_size} bytes")

                    self._track_free(ix, actual_size, options)
                    orig_free(opaque, size, options)

                device.allocator.alloc = wrap_alloc
                device.allocator.free = wrap_free
                self._orig_allocs[ix] = (orig_alloc, orig_free)
                self._wrapped.add(ix)

            return device

        Device.__getitem__ = wrap_getitem

    def _get_key(self, device: str, size: int, options: Optional[BufferSpec]) -> tuple:
        return (device, size, options)

    def _track_alloc(self, device: str, size: int, options: Optional[BufferSpec]):
        key = self._get_key(device, size, options)
        if key not in self.stats:
            self.stats[key] = AllocationStats(size, time.time(), time.time())
        self.stats[key].allocs += 1
        self.current += size
        self.total += size
        self.peak = max(self.peak, self.current)
        self.stats[key].last_used = time.time()

    def _track_hit(self, device: str, size: int, options: Optional[BufferSpec]):
        key = self._get_key(device, size, options)
        if key in self.stats:
            self.stats[key].hits += 1
            self.stats[key].reuses += 1
            self.stats[key].last_used = time.time()

    def _track_miss(self, device: str, size: int, options: Optional[BufferSpec]):
        key = self._get_key(device, size, options)
        if key in self.stats:
            self.stats[key].misses += 1

    def _track_free(self, device: str, size: int, options: Optional[BufferSpec]):
        key = self._get_key(device, size, options)
        if key in self.stats:
            self.stats[key].frees += 1
            self.current -= size

    def cleanup(self):
        Buffer.__init__ = self._orig_init
        for device_name, (orig_alloc, orig_free) in self._orig_allocs.items():
            try:
                device = self._orig_getitem(Device, device_name)
                device.allocator.alloc = orig_alloc
                device.allocator.free = orig_free
            except Exception:
                pass
        Device.__getitem__ = self._orig_getitem
        self._orig_allocs.clear()
        self._wrapped.clear()

    def print_summary(self):
        print("\nMemory Analysis Summary:")
        print(f"Total: {self.total / (1024*1024):.2f} MB")
        print(f"Peak: {self.peak / (1024*1024):.2f} MB")
        print(f"Current: {self.current / (1024*1024):.2f} MB\n")

        by_device = defaultdict(list)
        for (device, size, options), stats in self.stats.items():
            by_device[device].append((size, options, stats))

        print("device   buffer       size [B]   allocations   total [MB]")
        print("---------------------------------------------------------")
        row = "{:<6}   {:>6}   {:>12}   {:>11d}   {:>10.2f}"

        for device, allocations in by_device.items():
            if not allocations:
                continue
            for i, (size, options, stats) in enumerate(sorted(allocations, key=lambda x: x[0], reverse=True)):
                if stats.allocs == 0:
                    continue
                print(row.format(
                    device,
                    f"{i+1:d}",
                    size,
                    stats.allocs,
                    (size * stats.allocs) / (1024*1024)
                ))

    def __del__(self):
        self.cleanup()