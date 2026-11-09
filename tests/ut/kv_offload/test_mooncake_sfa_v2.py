# SPDX-License-Identifier: Apache-2.0
"""Persistent cache mapping, pipelined landing and PD-to-SFA request handoff."""

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
    MooncakePullRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import CopySfaTailDest
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import sparse_kv_offload_manager as manager_module
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
    CopySfaTopkSlotAllocator,
    copy_sfa_prefill_dest_geometry,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_v2 import (
    MooncakeSFAHostKV,
    MooncakeSFAPDState,
    MooncakeSFAStagingReader,
)
from vllm_ascend.kv_offload import dsa_pd_mooncake as legacy_registry

from .mooncake_v2.helpers import make_blocks, make_kv_cache_tensor, make_request, make_sfa_indexer_spec
from .mooncake_v2.test_pull_scheduler import make_pull_scheduler
from .test_dsa_pd_mooncake_connector_v2 import make_reader
from .test_sfa_pd_rd2h_connector import _make_layer, _make_tail_read_thread

MAIN = "model.layers.0.self_attn.attn"
INDEXER = "model.layers.0.self_attn.indexer.k_cache"


def make_manager(*, rank=0, size=2, blocks=4, k_dim=4):
    k = torch.zeros((blocks, 128, 1, k_dim), dtype=torch.bfloat16)
    v = torch.zeros((blocks, 128, 1, 2), dtype=torch.bfloat16)
    indexer = torch.zeros((blocks, 128, 1, 4), dtype=torch.int8)
    scales = torch.zeros((blocks, 128, 1, 1), dtype=torch.float16)
    main_spec = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=k_dim + 2, dtype=torch.bfloat16)
    indexer_spec = make_sfa_indexer_spec(block_size=128)
    hot_k = torch.zeros((4, 2304, 1, k_dim), dtype=torch.bfloat16)
    hot_v = torch.zeros((4, 2304, 1, 2), dtype=torch.bfloat16)
    caches = {
        INDEXER: (indexer, scales),
        MAIN: (torch.empty(0), torch.empty(0), k, v, hot_k, hot_v),
    }
    kv_config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=blocks * spec.page_size_bytes,
                layers=[name],
                layer_stride=blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
            for name, spec in [(INDEXER, indexer_spec), (MAIN, main_spec)]
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[INDEXER], kv_cache_spec=indexer_spec),
            KVCacheGroupSpec(layer_names=[MAIN], kv_cache_spec=main_spec),
        ],
    )
    manager = SimpleNamespace(
        tp_rank=rank,
        tp_group=SimpleNamespace(world_size=size),
        offload_layer_names=[MAIN],
        k_caches_cpu=[k],
        topk_buffers_k=[hot_k],
        topk_buffers_v=[hot_v],
        topk_buffer_size=2048,
        v_caches_cpu=[v],
        kv_cache_config=kv_config,
        _get_offload_layer_id=lambda name: 0,
        copy_sfa_kv=MagicMock(),
    )
    return manager, caches


@pytest.fixture(autouse=True)
def cpu_streams(monkeypatch):
    legacy_registry.reset_dsa_pd_mooncake_kv_registries()
    monkeypatch.setattr(torch.npu, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.npu, "Event", MagicMock())
    monkeypatch.setattr(torch.npu, "synchronize", MagicMock())
    yield
    legacy_registry.reset_dsa_pd_mooncake_kv_registries()


@pytest.mark.parametrize("rank", [0, 1])
def test_projection_uses_manager_views_excludes_scratch_and_keeps_legacy_pool_empty(rank):
    manager, caches = make_manager(rank=rank)
    if rank:
        # MRV2's non-owner tuple has placeholders; the manager restores its views.
        caches[MAIN] = (*caches[MAIN][:2], torch.empty(0), torch.empty(0), *caches[MAIN][4:])
    host = MooncakeSFAHostKV(manager, caches)
    assert host.transfer_caches[MAIN][0] is manager.k_caches_cpu[0]
    assert host.transfer_caches[MAIN][1] is manager.v_caches_cpu[0]
    assert host.transfer_caches[INDEXER] is caches[INDEXER]
    assert host.main_group_id == 1
    assert len(caches[MAIN]) == 6
    assert legacy_registry.get_dsa_pd_shared_host_pool() is None
    assert not legacy_registry.get_dsa_pd_mooncake_kv_registry()._offloaded_tensors


def test_v2_registration_publishes_only_persistent_components(monkeypatch):
    manager, caches = make_manager()
    monkeypatch.setattr(manager_module, "get_sparse_kv_offload_manager", lambda: manager)
    engine = MagicMock()
    monkeypatch.setattr(base_worker, "global_te", engine)
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.kv_cache_config = manager.kv_cache_config
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    worker.ascend_config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=1), sparse_kv_offload_config=SimpleNamespace(enabled=True)
    )
    worker.pcp_rank = worker.tp_rank = 0
    worker.tp_size = 2
    worker.engine_id, worker.te_rpc_port = "consumer", 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = 128, "127.0.0.1", 5000
    worker.register_kv_caches(caches)
    meta = worker.xfer_handshake_metadata
    assert meta.layer_names == [INDEXER, MAIN]
    assert meta.kv_caches_base_addr == [
        [tensor.data_ptr() for tensor in caches[INDEXER]],
        [manager.k_caches_cpu[0].data_ptr(), manager.v_caches_cpu[0].data_ptr()],
    ]
    assert meta.block_lens[1] == [128 * 4 * 2, 128 * 2 * 2]
    registered, lengths = engine.register_buffer.call_args.args
    for tensor in caches[INDEXER]:
        assert any(ptr <= tensor.data_ptr() < ptr + length for ptr, length in zip(registered, lengths))
    for tensor in caches[MAIN][4:]:
        assert any(ptr <= tensor.data_ptr() < ptr + length for ptr, length in zip(registered, lengths))
    for tensor in caches[MAIN][2:4]:
        assert not any(ptr <= tensor.data_ptr() < ptr + length for ptr, length in zip(registered, lengths))


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_main_blocks_are_partitioned_once_while_indexer_stays_replicated(tp_size):
    all_main = []
    for rank in range(tp_size):
        manager, caches = make_manager(rank=rank, size=tp_size)
        host = MooncakeSFAHostKV(manager, caches)
        thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
        thread._shared_host_pool = host
        ownership = {}
        for tensor in host.transfer_caches[MAIN]:
            src, dst, lengths = [], [], []
            thread._append_block_transfer_addresses(
                local_block_groups=[[0, 1, 2, 3]],
                remote_block_groups=[[4, 5, 6, 7]],
                local_base_addr=tensor.data_ptr(),
                remote_base_addr=0x10000000,
                local_block_stride=16,
                remote_block_stride=16,
                transfer_len=16,
                src_list=src,
                dst_list=dst,
                length_list=lengths,
                shared_block_groups=ownership,
                shared_host_pool=host,
            )
            all_main.append(sum(lengths))
        assert len(ownership) == 1
        src, dst, lengths = [], [], []
        thread._append_block_transfer_addresses(
            local_block_groups=[[0, 1, 2, 3]],
            remote_block_groups=[[4, 5, 6, 7]],
            local_base_addr=caches[INDEXER][0].data_ptr(),
            remote_base_addr=0x20000000,
            local_block_stride=16,
            remote_block_stride=16,
            transfer_len=16,
            src_list=src,
            dst_list=dst,
            length_list=lengths,
            shared_block_groups=ownership,
            shared_host_pool=host,
        )
        assert lengths == [64]
    assert sum(all_main) == 2 * 4 * 16


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("metadata_targets", [False, True])
def test_real_bytes_land_through_manager_with_existing_pipeline(monkeypatch, pipeline, metadata_targets):
    manager, caches = make_manager(blocks=4, k_dim=8192)
    host = MooncakeSFAHostKV(manager, caches)
    engine = MagicMock()
    reader = MooncakeSFAStagingReader.__new__(MooncakeSFAStagingReader)
    reader.__dict__.update(make_reader(engine, pipeline=pipeline, window=4).__dict__)
    reader.host_kv, reader.kv_caches = host, host.transfer_caches
    tensors = [*host.transfer_caches[MAIN], *host.transfer_caches[INDEXER]]
    reader.block_len_per_addr = [[128 * 8192 * 2], [128 * 4]]
    sources = [torch.full((tensor.nbytes,), i + 1, dtype=torch.int8) for i, tensor in enumerate(tensors)]
    reads = []

    def read(_session, local, remote, lengths):
        reads.extend(zip(remote, lengths))
        for dst, src, length in zip(local, remote, lengths):
            ctypes.memmove(dst, src, length)
        return 0

    def land(src, dst, lengths, count):
        assert src.dtype == dst.dtype == torch.int64
        assert lengths.dtype == count.dtype == torch.int32
        for a, b, length in zip(src.tolist(), dst.tolist(), lengths.tolist()):
            ctypes.memmove(b, a, length)

    engine.batch_transfer_sync_read.side_effect = read
    manager.copy_sfa_kv.side_effect = land
    kwargs = {}
    if metadata_targets:
        table = reader.get_transfer_targets(
            [MAIN, INDEXER], [[t.data_ptr() for t in host.transfer_caches[n]] for n in [MAIN, INDEXER]]
        )
        kwargs["transfer_targets"] = [*table[0], *table[1]]
    assert (
        reader.batch_transfer_sync_read(
            "peer",
            [t.data_ptr() for t in tensors],
            [t.data_ptr() for t in sources],
            [t.nbytes for t in tensors],
            "request",
            **kwargs,
        )
        == 0
    )
    assert sum(length for _, length in reads) == sum(t.nbytes for t in tensors)
    for source, tensor in zip(sources, tensors):
        assert torch.equal(source, tensor.view(torch.int8).reshape(-1))
    assert manager.copy_sfa_kv.called
    assert all(
        not bank.completion_pending and not bank.inflight_owners for bank in reader._dsa_pd_mooncake_staging_banks
    )


@pytest.mark.parametrize("tokens", [0, 1, 127, 128, 129, 2047, 2048, 2049, 2176, 2177])
def test_first_decode_preparation_is_local_and_once_per_request(tokens):
    state = MooncakeSFAPDState(main_group_id=1)
    state.update({"r": 2}, ["r"])
    with pytest.raises(RuntimeError, match="receive completed"):
        state.prepare(["r"], [2], [tokens], block_size=128, hot_tokens=2048)
    state.record_received({"r"}, set())
    restore, dense, initialized = state.prepare(["r"], [2], [tokens], block_size=128, hot_tokens=2048)
    assert restore is False
    assert dense == {}
    assert initialized == {"r"}
    # Preparation is committed only after metadata/copy submission succeeds.
    assert state.pending == {"r"}
    state.commit(initialized)
    assert state.prepare(["r"], [2], [tokens + 1], block_size=128, hot_tokens=2048) == (False, {}, set())


def test_receive_failure_cancellation_and_row_reuse_do_not_publish_stale_readiness():
    state = MooncakeSFAPDState(0)
    state.update({"old": 0}, ["old"])
    state.record_received({"old"}, {"old"})
    assert not state.received
    state.update({}, [])
    state.record_received({"old"}, set())
    state.update({"new": 0}, ["new"])
    assert state.pending == {"new"} and not state.received
    with pytest.raises(RuntimeError, match="receive completed"):
        state.prepare(["new"], [0], [128], block_size=128, hot_tokens=2048)


def test_complete_local_prefix_hit_initializes_its_new_row_without_read():
    state = MooncakeSFAPDState(0)
    state.update({"hit": 1}, [])
    assert state.prepare(["hit"], [1], [128], block_size=128, hot_tokens=2048) == (False, {1: (0, 128)}, {"hit"})


def test_scheduler_reserves_defers_and_releases_existing_sfa_slots():
    scheduler = make_pull_scheduler()
    scheduler._copy_sfa_slot_allocator = CopySfaTopkSlotAllocator(1)
    request = make_request(
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_block_ids": ([4, 5],),
            "remote_engine_id": "p",
            "remote_host": "127.0.0.1",
            "remote_port": 5000,
            "remote_request_id": "p-r",
            "remote_num_prompt_tokens": 32,
        }
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, True)
    scheduler.update_state_after_alloc(request, make_blocks(), 32)
    meta = scheduler.build_connector_meta(MagicMock())
    assert meta.copy_sfa_slots == {request.request_id: 0}
    assert meta.requests[request.request_id].copy_sfa_slot == 0
    waiter = make_request(request_id="waiter", kv_transfer_params={"do_remote_prefill": True})
    assert scheduler.get_num_new_matched_tokens(waiter, 0) == (None, False)
    scheduler.request_finished(request, ([0, 1],))
    assert scheduler.get_num_new_matched_tokens(waiter, 0) == (None, False)
    scheduler._recving_thread = MagicMock()
    scheduler.update_connector_output(
        SimpleNamespace(finished_recving={request.request_id}, failed_recving=set(), finished_sending=None)
    )
    assert scheduler.get_num_new_matched_tokens(waiter, 0) == (32, True)


def test_early_mrv2_metadata_preparation_is_idempotent_with_late_start_load():
    worker = MooncakePullConnectorWorker.__new__(MooncakePullConnectorWorker)
    worker._sfa_pd_state = MooncakeSFAPDState(0)
    worker._recving_thread = MagicMock()
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    meta = MooncakeConnectorMetadata()
    meta.copy_sfa_slots = {"hit": 0}
    facade = MooncakePullConnector.__new__(MooncakePullConnector)
    facade.connector_worker = worker
    facade.prepare_sparse_kv_offload(meta)
    assert facade.get_copy_sfa_slot_bindings() == {"hit": 0}
    worker._sfa_pd_state.commit({"hit"})
    worker.start_load_kv(meta)
    assert not worker._sfa_pd_state.pending
    worker._recving_thread.add_requests.assert_not_called()


@pytest.mark.parametrize("tokens", [0, 1, 127, 128, 129, 2047, 2048, 2049, 2176, 2177, 2303, 2304, 4095, 4096, 4097])
@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_direct_hbm_reads_match_unchanged_sfa_dense_and_circular_tail(tokens, tp_size):
    blocks = list(range(8, 16)) + list(range(8)) + list(range(16, 40))
    for rank in range(tp_size):
        manager, caches = make_manager(rank=rank, size=tp_size, blocks=40)
        host = MooncakeSFAHostKV(manager, caches)
        pair = host.hot_caches[MAIN]
        request = SimpleNamespace(
            copy_sfa_slot=3,
            num_computed_tokens=0,
            num_external_tokens=tokens + 1,
            local_num_prompt_tokens=tokens + 1,
            remote_num_prompt_tokens=tokens + 1,
            remote_block_ids=([], blocks),
        )
        local, peer, lengths, targets = [], [], [], []
        host.append_prefill_reads(MAIN, request, (1000, 2000), (1024, 512), (1024, 512), local, peer, lengths, targets)
        thread = _make_tail_read_thread(tp_rank=rank, tp_size=tp_size)
        dense, tail_tokens, tail_block = copy_sfa_prefill_dest_geometry(tokens, 128, 2048)
        thread._state.copy_sfa_tail_by_req["req-0"] = CopySfaTailDest(3, tail_tokens, tail_block, dense, tokens)
        thread._state.dest_blocks_by_req["req-0"] = (list(range(40)), [])
        thread._state.topk_k_bases = [pair[0].data_ptr()]
        thread._state.topk_v_bases = [pair[1].data_ptr()]
        thread._state.topk_row_tokens = 2304
        thread._state.topk_hot_tokens = 2048
        layer = _make_layer(k_cpu_ptr=None, v_cpu_ptr=None, has_indexer=False)
        layer["p_k_len"], layer["p_v_len"] = 1024, 512
        expected_local, expected_peer, expected_lengths, _ = thread._build_req_descriptors(
            layer, "req-0", p_main_block_ids=blocks, p_indexer_block_ids=[], want_info=False
        )
        assert (local, peer, lengths) == (expected_local, expected_peer, expected_lengths)
        assert len(targets) == len(lengths)
        assert all(not target.offloaded for target in targets)
        assert sum(lengths) == (tokens if dense else tail_tokens) * 12


@pytest.mark.parametrize("tokens", [129, 2049, 2177, 4097])
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("rank,size", [(0, 1), (7, 8)])
@pytest.mark.parametrize("fail_direct", [False, True])
def test_real_bucket_direct_hbm_and_host_landing_share_completion(tokens, pipeline, rank, size, fail_direct):
    manager, caches = make_manager(rank=rank, size=size, blocks=40)
    host = MooncakeSFAHostKV(manager, caches)
    engine = MagicMock()
    reader = MooncakeSFAStagingReader.__new__(MooncakeSFAStagingReader)
    reader.__dict__.update(make_reader(engine, pipeline=pipeline, window=4).__dict__)
    reader.host_kv, reader.kv_caches = host, host.transfer_caches
    reader.block_len_per_addr = [[1024, 512]]
    source_k = torch.arange(40 * 128 * 4, dtype=torch.float32).to(torch.bfloat16).reshape(40, 128, 1, 4)
    source_v = torch.arange(40 * 128 * 2, dtype=torch.float32).neg().to(torch.bfloat16).reshape(40, 128, 1, 2)
    reads = []

    def read(_session, local, remote, lengths):
        reads.extend(zip(local, remote, lengths))
        if fail_direct:
            return -1
        for dst, src, length in zip(local, remote, lengths):
            ctypes.memmove(dst, src, length)
        return 0

    def land(src, dst, lengths, count):
        assert lengths.dtype == count.dtype == torch.int32
        for a, b, length in zip(src.tolist(), dst.tolist(), lengths.tolist()):
            ctypes.memmove(b, a, length)

    engine.batch_transfer_sync_read.side_effect = read
    manager.copy_sfa_kv.side_effect = land
    count = (tokens + 127) // 128
    blocks = list(range(8, 16)) + list(range(8)) + list(range(16, 40))
    request = SimpleNamespace(
        copy_sfa_slot=2,
        num_computed_tokens=0,
        num_external_tokens=tokens + 1,
        local_num_prompt_tokens=tokens + 1,
        remote_num_prompt_tokens=tokens + 1,
        remote_block_ids=([], blocks),
    )
    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    thread.staging_reader, thread.engine, thread._shared_host_pool = reader, engine, host
    thread._transfer_targets = [host.registry.create_transfer_targets([list(host.transfer_caches[MAIN])])[0]]
    thread.kv_cache_specs = [manager.kv_cache_config.kv_cache_groups[1].kv_cache_spec]
    thread.layer_names = [MAIN]
    thread.kv_caches_base_addr = [[t.data_ptr() for t in host.transfer_caches[MAIN]]]
    thread.block_strides = thread.block_lens = [[1024, 512]]
    thread.block_shapes = [[(128, 1, 4), (128, 1, 2)]]
    thread.tp_rank, thread.tp_size = rank, size
    thread.local_metadata = SimpleNamespace(local_ip="local")
    peer = SimpleNamespace(
        local_ip="peer", te_rpc_port=9, kv_caches_base_addr=[[source_k.data_ptr(), source_v.data_ptr()]]
    )
    metadata = SimpleNamespace(
        metadata_by_pcp_rank={0: SimpleNamespace(metadata_by_tp_rank={0: peer})},
        block_strides=[[1024, 512]],
        block_lens=[[1024, 512]],
    )
    args = (0, 0, 0, 1, 1, 1, metadata, {0: {(0, 0): [("r", list(range(count)), blocks[:count])]}})
    if fail_direct:
        with pytest.raises(RuntimeError, match="KV transfer failed"):
            thread._execute_tp_transfer_bucket(*args, sfa_requests={"r": request})
        manager.copy_sfa_kv.assert_not_called()
        return
    thread._execute_tp_transfer_bucket(*args, sfa_requests={"r": request})
    dense = tokens <= 2048
    tail = tokens % 128
    start = 0 if dense else 2048 + ((tokens // 128) % 2) * 128
    length = tokens if dense else tail
    for source, hot in zip((source_k, source_v), host.hot_caches[MAIN]):
        expected = source[blocks].reshape(-1, 1, source.shape[-1])[:tokens]
        if not dense:
            expected = expected[-tail:]
        assert torch.equal(hot[2, start : start + length], expected)
        assert not torch.count_nonzero(hot[:2])
        assert not torch.count_nonzero(hot[3:])
    hot_bytes = sum(
        length
        for dst, _, length in reads
        if any(t.data_ptr() <= dst < t.data_ptr() + t.nbytes for t in host.hot_caches[MAIN])
    )
    assert hot_bytes == length * 12
    assert all(not bank.completion_pending for bank in getattr(reader, "_dsa_pd_mooncake_staging_banks", []))


def test_remote_pd_prefix_must_match_first_decode_and_cancel_releases_it():
    state = MooncakeSFAPDState(1)
    state.update({"r": 0}, ["r"], {"r": 2175})
    state.record_received({"r"}, set())
    with pytest.raises(RuntimeError, match="prefix differs"):
        state.prepare(["r"], [0], [2176], block_size=128, hot_tokens=2048)
    assert state.prepare(["r"], [0], [2175], block_size=128, hot_tokens=2048) == (False, {}, {"r"})
    state.update({}, [])
    assert not state.remote and not state.remote_prefixes


def test_partial_local_prefix_still_primes_whole_dense_row_from_producer():
    manager, caches = make_manager()
    host = MooncakeSFAHostKV(manager, caches)
    req = SimpleNamespace(
        copy_sfa_slot=0,
        num_computed_tokens=128,
        num_external_tokens=73,
        local_num_prompt_tokens=201,
        remote_num_prompt_tokens=201,
        remote_block_ids=([], [5, 9]),
    )
    local, peer, lengths = [], [], []
    host.append_prefill_reads(MAIN, req, (1000, 2000), (1024, 512), (1024, 512), local, peer, lengths)
    assert peer[0] == 1000 + 5 * 1024
    assert sum(lengths) == 200 * 12


@pytest.mark.parametrize("slot", [None, -1, 4])
def test_direct_hot_reads_require_valid_reserved_row(slot):
    manager, caches = make_manager()
    host = MooncakeSFAHostKV(manager, caches)
    req = SimpleNamespace(
        copy_sfa_slot=slot,
        num_computed_tokens=0,
        num_external_tokens=130,
        local_num_prompt_tokens=130,
        remote_num_prompt_tokens=130,
        remote_block_ids=([], [0, 1]),
    )
    with pytest.raises(ValueError, match="hot row"):
        host.append_prefill_reads(MAIN, req, (1000, 2000), (1024, 512), (1024, 512), [], [], [])


@pytest.mark.parametrize("failed", [False, True])
def test_cancelled_pd_holds_hot_row_until_all_receivers_finish(failed):
    scheduler = make_pull_scheduler()
    scheduler._copy_sfa_slot_allocator = CopySfaTopkSlotAllocator(1)
    scheduler._copy_sfa_slot_allocator.bind("cancelled")
    scheduler._reqs_recv_info["cancelled"] = ("peer", 9, "p-r")
    scheduler._recving_thread = MagicMock()
    scheduler.kv_transfer_config = SimpleNamespace(kv_load_failure_policy="fail")
    request = make_request(request_id="cancelled", kv_transfer_params={"do_remote_prefill": False})
    scheduler.request_finished(request, ([0],))
    assert scheduler._copy_sfa_slot_allocator.bound_slots() == {"cancelled": 0}
    assert not scheduler._copy_sfa_slot_allocator.can_bind("next")
    scheduler.update_connector_output(
        SimpleNamespace(
            finished_recving={"cancelled"}, failed_recving={"cancelled"} if failed else set(), finished_sending=None
        )
    )
    assert not scheduler._copy_sfa_deferred_releases
    assert scheduler._copy_sfa_slot_allocator.bind("next") == 0
