from collections import defaultdict
from dataclasses import dataclass
import json
import os
import time
from typing import Dict, Optional, Any
from tinygrad.device import LRUAllocator, BufferSpec, Device

@dataclass
class AllocationStats:
    size: int
    creation_time: float
    last_used: float
    hits_in_cache: int = 0
    misses_from_cache: int = 0
    allocations: int = 0
    frees: int = 0
    cache_reuses: int = 0

    def to_dict(self):
        return {
            "size_bytes": self.size,
            "size_kb": self.size / 1024,
            "creation_time": self.creation_time,
            "last_used": self.last_used,
            "hits_in_cache": self.hits_in_cache,
            "misses_from_cache": self.misses_from_cache,
            "allocations": self.allocations,
            "frees": self.frees,
            "cache_reuses": self.cache_reuses,
            "cache_reuse_rate": (self.cache_reuses / self.allocations * 100) if self.allocations else 0
        }

class MemoryAnalyzer:
    def __init__(self):
        # Cleanup any existing monitoring
        if hasattr(self, 'original_allocators'):
            self.cleanup()

        self.original_allocators = {}
        self.stats = {}
        self.total_allocated = 0
        self.peak_allocated = 0
        self.current_allocated = 0

        # Only wrap devices when they're actually opened
        self._wrapped_devices = set()
        self._original_getitem = Device.__getitem__

        def wrapped_getitem(device_self, ix:str):
            device = self._original_getitem(device_self, ix)
            # Only wrap if it's a new device with an LRU allocator
            device_name = device.device
            if device_name not in self._wrapped_devices and isinstance(device.allocator, LRUAllocator):
                orig_alloc = device.allocator.alloc
                orig_free = device.allocator.free

                def wrapped_alloc(size: int, options: Optional[BufferSpec] = None, dev=device_name):
                    key = (dev, size, options)
                    # Convert size to bytes explicitly
                    actual_size = size
                    if options and hasattr(options, 'dtype'):
                        actual_size *= options.dtype.itemsize

                    if len(device.allocator.cache.get((size, options), [])):
                        self._track_cache_hit(dev, actual_size, options)
                        return orig_alloc(size, options)  # Let original allocator handle cache

                    self._track_cache_miss(dev, actual_size, options)
                    result = orig_alloc(size, options)
                    self._track_allocation(dev, actual_size, options)
                    return result

                def wrapped_free(opaque: Any, size: int, options: Optional[BufferSpec] = None, dev=device_name):
                    self._track_free(dev, size, options)
                    orig_free(opaque, size, options)

                device.allocator.alloc = wrapped_alloc
                device.allocator.free = wrapped_free
                self.original_allocators[device_name] = (orig_alloc, orig_free)
                self._wrapped_devices.add(device_name)
            return device

        # Monkey patch Device.__getitem__
        Device.__getitem__ = wrapped_getitem

    def _get_stats_key(self, device: str, size: int, options: Optional[BufferSpec]) -> tuple[str, int, Optional[BufferSpec]]:
        return (device, size, options)

    def _track_allocation(self, device: str, size: int, options: Optional[BufferSpec]):
        key = self._get_stats_key(device, size, options)
        if key not in self.stats:
            self.stats[key] = AllocationStats(size, time.time(), time.time())
        self.stats[key].allocations += 1
        self.current_allocated += size
        self.total_allocated += size
        self.peak_allocated = max(self.peak_allocated, self.current_allocated)
        self.stats[key].last_used = time.time()

    def _track_cache_hit(self, device: str, size: int, options: Optional[BufferSpec]):
        key = self._get_stats_key(device, size, options)
        if key in self.stats:
            self.stats[key].hits_in_cache += 1
            self.stats[key].cache_reuses += 1
            self.stats[key].last_used = time.time()

    def _track_cache_miss(self, device: str, size: int, options: Optional[BufferSpec]):
        key = self._get_stats_key(device, size, options)
        if key in self.stats:
            self.stats[key].misses_from_cache += 1

    def _track_free(self, device: str, size: int, options: Optional[BufferSpec]):
        key = self._get_stats_key(device, size, options)
        if key in self.stats:
            self.stats[key].frees += 1
            self.current_allocated -= size

    def cleanup(self):
        """Restore original allocators and device access"""
        for device_name, (orig_alloc, orig_free) in self.original_allocators.items():
            try:
                device = self._original_getitem(Device, device_name)
                device.allocator.alloc = orig_alloc
                device.allocator.free = orig_free
            except Exception:
                pass  # Device might not exist anymore, that's okay

        # Restore original Device.__getitem__
        if hasattr(self, '_original_getitem'):
            Device.__getitem__ = self._original_getitem

        self.original_allocators.clear()
        self._wrapped_devices.clear()

    def print_summary(self):
        """Print a summary of memory usage statistics"""
        print("\n=== Memory Analysis Summary ===")
        print(f"Total Allocated: {self.total_allocated / (1024*1024):.2f} MB")
        print(f"Peak Allocated: {self.peak_allocated / (1024*1024):.2f} MB")
        print(f"Current Allocated: {self.current_allocated / (1024*1024):.2f} MB")

        print("\nAllocation Analysis by Device and Size:")
        by_device = defaultdict(list)
        for (device, size, options), stats in self.stats.items():
            by_device[device].append((size, options, stats))

        for device, allocations in by_device.items():
            total_device_allocs = sum(stats.allocations for _, _, stats in allocations)
            total_device_bytes = sum(size * stats.allocations for size, _, stats in allocations)
            print(f"\nDevice: {device}")
            print(f"Total Allocations: {total_device_allocs}")
            print(f"Total Memory Throughput: {total_device_bytes / (1024*1024):.2f} MB")

            for size, options, stats in sorted(allocations, key=lambda x: x[2].allocations, reverse=True):
                if stats.allocations == 0: continue  # Skip unused sizes
                print(f"\nBuffer Size: {size / 1024:.2f} KB {options if options else ''}")
                print(f"  Total Allocations: {stats.allocations}")
                print(f"  Active Allocations: {stats.allocations - stats.frees}")
                print(f"  Cache Hits/Misses: {stats.hits_in_cache}/{stats.misses_from_cache}")
                print(f"  Cache Hit Rate: {(stats.hits_in_cache / (stats.hits_in_cache + stats.misses_from_cache) * 100) if (stats.hits_in_cache + stats.misses_from_cache) else 0:.1f}%")
                print(f"  Total Bytes: {(size * stats.allocations) / (1024*1024):.2f} MB")

    def write_json(self, output_path: str):
        """Write detailed analysis to a JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        analysis = {
            "summary": {
                "total_allocated_bytes": self.total_allocated,
                "total_allocated_mb": self.total_allocated / (1024*1024),
                "peak_allocated_bytes": self.peak_allocated,
                "peak_allocated_mb": self.peak_allocated / (1024*1024),
                "current_allocated_bytes": self.current_allocated,
                "current_allocated_mb": self.current_allocated / (1024*1024)
            },
            "devices": defaultdict(list)
        }

        for (device, size, options), stats in self.stats.items():
            analysis["devices"][device].append({
                "buffer_spec": str(options) if options else "default",
                **stats.to_dict()
            })

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

    def __del__(self):
        """Ensure cleanup when the analyzer is destroyed"""
        self.cleanup()