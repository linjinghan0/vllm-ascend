# SPDX-License-Identifier: Apache-2.0
import random
from unittest import TestCase

from vllm_ascend.kv_offload.dsa_pd_shared_host import DSAPDSharedHostPool, partition_shared_block_groups


class TestSharedHostOwnership(TestCase):
    def test_exact_coverage_for_fragmented_uneven_and_empty_requests(self):
        rng = random.Random(19212)
        for size in (1, 2, 3, 4, 8, 16):
            for count in (0, 1, 2, 7, 8, 9, 31, 32, 33, 97):
                local = rng.sample(range(2048), count)
                remote = rng.sample(range(4096), count)
                # Fragmentation and grouping must not change ownership or
                # accidentally pair a local page with the next source page.
                groups = [local[i : i + 3] for i in range(0, count, 3)]
                peers = [remote[i : i + 3] for i in range(0, count, 3)]
                owned = []
                for rank in range(size):
                    dst, src = partition_shared_block_groups(groups, peers, rank, size)
                    pairs = [(a, b) for x, y in zip(dst, src) for a, b in zip(x, y)]
                    self.assertTrue(all(a % size == rank for a, _ in pairs))
                    owned.extend(pairs)
                self.assertCountEqual(owned, list(zip(local, remote)))
                self.assertEqual(len(owned), len(set(owned)))

    def test_coalesces_only_when_source_and_destination_are_both_contiguous(self):
        self.assertEqual(
            partition_shared_block_groups([[1, 2], [3], [7, 8]], [[10, 11], [12], [20, 25]], 0, 1),
            ([[1, 2, 3], [7], [8]], [[10, 11, 12], [20], [25]]),
        )

    def test_invalid_ownership_or_mismatched_descriptors_fail_before_transfer(self):
        for local, remote, rank, size in (
            ([[1]], [], 0, 1),
            ([[1, 2]], [[3]], 0, 1),
            ([[1]], [[-1]], 0, 1),
            ([], [], -1, 2),
            ([], [], 2, 2),
            ([], [], 0, 0),
        ):
            with self.assertRaises(ValueError):
                partition_shared_block_groups(local, remote, rank, size)

    def test_shared_pointer_lookup_rejects_padding_crossings_and_foreign_buffers(self):
        pool = object.__new__(DSAPDSharedHostPool)
        pool._starts = [1024, 4096]
        pool._ends = [2048, 8192]
        self.assertTrue(pool.contains(1024, 1024))
        self.assertTrue(pool.contains(8191, 1))
        for ptr, length in ((1023, 1), (2048, 1), (2047, 2), (3072, 1), (8191, 2), (1024, 0)):
            self.assertFalse(pool.contains(ptr, length))

    def test_shared_allocator_may_return_nonmonotonic_nonoverlapping_addresses(self):
        pool = object.__new__(DSAPDSharedHostPool)
        pool._starts, pool._ends = [], []
        pool._register_range(4096, 1024)
        pool._register_range(1024, 1024)
        pool._register_range(2048, 1024)
        self.assertEqual(pool._starts, [1024, 2048, 4096])
        self.assertTrue(pool.contains(4096, 1024))
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            pool._register_range(4000, 200)
