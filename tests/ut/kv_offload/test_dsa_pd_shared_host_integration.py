# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ascend_config import DSASparseAttentionConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.kv_offload.dsa_pd_shared_host import DSAPDSharedHostPool


class TestSharedHostIntegration(TestCase):
    def test_shared_host_ownership_is_applied_before_address_building(self):
        pool = SimpleNamespace(rank=3, size=8, contains=lambda ptr: ptr == 1024)
        local, remote, lengths = [], [], []
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.get_dsa_pd_shared_host_pool",
            return_value=pool,
        ):
            MooncakePullRecvingThread._append_block_transfer_addresses(
                [list(range(16))], [list(range(20, 36))], 1024, 100000, 128, 128, 128, local, remote, lengths
            )
        self.assertEqual(local, [1024 + 3 * 128, 1024 + 11 * 128])
        self.assertEqual(remote, [100000 + 23 * 128, 100000 + 31 * 128])
        self.assertEqual(lengths, [128, 128])

    def test_rank_local_hbm_destinations_keep_the_complete_transfer(self):
        pool = SimpleNamespace(rank=3, size=8, contains=lambda ptr: False)
        local, remote, lengths = [], [], []
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.get_dsa_pd_shared_host_pool",
            return_value=pool,
        ):
            MooncakePullRecvingThread._append_block_transfer_addresses(
                [list(range(16))], [list(range(20, 36))], 1024, 100000, 128, 128, 128, local, remote, lengths
            )
        self.assertEqual((local, remote, lengths), ([1024], [100000 + 20 * 128], [16 * 128]))

    def test_non_owner_slots_are_masked_and_reused_without_modifying_metadata(self):
        pool = object.__new__(DSAPDSharedHostPool)
        pool.rank = 1
        pool._masked_slots = {}
        slots = torch.tensor([7, 20, -1], dtype=torch.int32)
        masked = pool.write_slots(slots)
        self.assertEqual(masked.tolist(), [-1, -1, -1])
        self.assertEqual(slots.tolist(), [7, 20, -1])
        self.assertIs(pool.write_slots(slots + 1), masked)
        pool.rank = 0
        self.assertIs(pool.write_slots(slots), slots)

    def test_write_fence_orders_all_ranks_and_retains_the_collective_result(self):
        pool = object.__new__(DSAPDSharedHostPool)
        pool.group = MagicMock()
        pool._fence = object()
        old = pool._fence
        pool.fence_writes()
        pool.group.all_reduce.assert_called_once_with(old)
        self.assertIs(pool._fence, pool.group.all_reduce.return_value)

    def test_landing_keeps_descriptors_alive_even_on_a_native_error(self):
        pool = object.__new__(DSAPDSharedHostPool)
        pool.device = torch.device("cpu")
        pool._starts, pool._ends = [1024], [4096]
        pool._offload = SimpleNamespace(sparse_copy=MagicMock(return_value=-1))
        staging = torch.zeros(64, dtype=torch.int8)
        copies = [
            SimpleNamespace(destination=1056, length=16, staging_offset=8),
            SimpleNamespace(destination=1088, length=32, staging_offset=24),
        ]
        owners = []
        with self.assertRaisesRegex(RuntimeError, "landing failed"):
            pool.land(copies, staging, owners)
        self.assertEqual(len(owners), 1)
        addresses, lengths, count = owners[0]
        self.assertEqual(addresses.tolist(), [[staging.data_ptr() + 8, staging.data_ptr() + 24], [1056, 1088]])
        self.assertEqual(lengths.tolist(), [16, 32])
        self.assertEqual(addresses.dtype, torch.int64)
        self.assertEqual(lengths.dtype, torch.int32)
        self.assertEqual(count.dtype, torch.int32)
        self.assertEqual(count.tolist(), [2])

    def test_out_of_bounds_landing_is_rejected_before_launching_a_copy(self):
        pool = object.__new__(DSAPDSharedHostPool)
        pool._starts, pool._ends = [1024], [4096]
        pool._offload = SimpleNamespace(sparse_copy=MagicMock())
        with self.assertRaisesRegex(ValueError, "outside"):
            pool.land([SimpleNamespace(destination=4090, length=16)], torch.zeros(16), [])
        pool._offload.sparse_copy.assert_not_called()

    def test_landing_rejects_length_overflow_before_constructing_native_descriptors(self):
        pool = object.__new__(DSAPDSharedHostPool)
        pool._starts, pool._ends = [1024], [1 << 33]
        pool._offload = SimpleNamespace(sparse_copy=MagicMock())
        with self.assertRaisesRegex(ValueError, "int32 ABI"):
            pool.land([SimpleNamespace(destination=1024, length=1 << 31)], torch.zeros(1), [])
        pool._offload.sparse_copy.assert_not_called()

    def test_shared_configuration_defaults_and_invalid_combinations(self):
        self.assertFalse(DSASparseAttentionConfig({}, None).tp_shared_host_kv)
        for config in (
            dict(tp_shared_host_kv=True),
            dict(mode="fused_overlap", tp_shared_host_kv=True),
            dict(shared_host_store_port=0),
            dict(shared_host_store_port=True),
            dict(tp_shared_host_kv="not-a-bool"),
        ):
            with self.assertRaises(ValueError):
                DSASparseAttentionConfig(config, None)
