# SPDX-License-Identifier: Apache-2.0
"""Deterministic bank-worker overlap, byte coverage, and join/drain regressions."""

import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend import envs
from vllm_ascend.kv_offload import dsa_pd_mooncake as registry
from vllm_ascend.kv_offload import dsa_pd_mooncake_staging as staging_module

from .test_dsa_pd_mooncake_connector_v2 import make_reader

ALIGNMENT = 2 * 1024 * 1024
WINDOW = 16
BANKS = 4
PIECE_BYTES = 128
HOST_STRIDE = 192


@pytest.fixture(autouse=True)
def reset_registry():
    registry.reset_dsa_pd_mooncake_kv_registries()
    yield
    registry.reset_dsa_pd_mooncake_kv_registries()


class BankRuntime:
    def __init__(self, monkeypatch, reader, workers, batches):
        self.reader = reader
        self.workers = workers
        self.lock = threading.Lock()
        self.local = threading.local()
        self.threads = set()
        self.reads = []
        self.active_reads = 0
        self.peak_reads = 0
        self.bank_owners = set()
        self.read_hook = None
        self.copy_hook = None
        self.event_hook = None
        self.barrier = threading.Barrier(min(workers, BANKS, batches)) if workers else None
        self.queues = [[] for _ in range(BANKS)]
        self.copy = reader._copy_dsa_pd_mooncake_staging_to_offloaded_kv
        runtime = self

        class Stream:
            def __init__(self, index):
                self.index = index

            def synchronize(self):
                assert reader._dsa_pd_mooncake_staging_lock.locked()
                for task in tuple(runtime.queues[self.index]):
                    runtime.queues[self.index].remove(task)
                    task()

        class Event:
            def record(self, stream):
                self.stream = stream
                self.tasks = tuple(runtime.queues[stream.index])
                if runtime.event_hook is not None:
                    runtime.event_hook(stream.index)

            def synchronize(self):
                assert reader._dsa_pd_mooncake_staging_lock.locked()
                for task in self.tasks:
                    if task in runtime.queues[self.stream.index]:
                        runtime.queues[self.stream.index].remove(task)
                        task()

        def set_device(device):
            self.local.device = device
            with self.lock:
                self.threads.add(threading.current_thread())

        @contextmanager
        def stream_context(stream):
            assert getattr(self.local, "stream", None) is None
            self.local.stream = stream
            try:
                yield
            finally:
                self.local.stream = None

        def submit_copy(*args, **kwargs):
            bank = kwargs["bank"]
            assert self.local.stream is bank.stream
            self.queues[bank.stream.index].append(lambda: self.copy(*args, **kwargs))
            if self.copy_hook is not None:
                self.copy_hook(bank.stream.index)

        execute = reader._execute_dsa_pd_mooncake_bank_batch

        def owned_execute(session, bank, batch, *args):
            assert reader._dsa_pd_mooncake_staging_lock.locked()
            if self.workers:
                assert self.local.device == reader._dsa_pd_mooncake_staging_tensor.device
            with self.lock:
                assert batch.bank_index not in self.bank_owners
                self.bank_owners.add(batch.bank_index)
            try:
                return execute(session, bank, batch, *args)
            finally:
                with self.lock:
                    self.bank_owners.remove(batch.bank_index)

        ensure = reader._ensure_dsa_pd_mooncake_staging_tensor

        def frozen_allocation():
            assert not threading.current_thread().name.startswith("dsa-mooncake-bank")
            return ensure()

        reader._landing_streams = [Stream(index) for index in range(BANKS)]
        reader._landing_stream = reader._landing_streams[0]
        reader._copy_dsa_pd_mooncake_staging_to_offloaded_kv = submit_copy
        reader._execute_dsa_pd_mooncake_bank_batch = owned_execute
        reader._ensure_dsa_pd_mooncake_staging_tensor = frozen_allocation
        monkeypatch.setattr(torch.npu, "set_device", set_device)
        monkeypatch.setattr(torch.npu, "stream", stream_context)
        monkeypatch.setattr(torch.npu, "Event", Event)
        monkeypatch.setattr(torch.npu, "synchronize", MagicMock())
        reader.engine.batch_transfer_sync_read.side_effect = self.read

    def read(self, session, local, remote, lengths):
        staging_ptr = self.reader._dsa_pd_mooncake_staging_tensor.data_ptr()
        bank_index = (local[0] - staging_ptr) // (ALIGNMENT * (WINDOW // BANKS))
        with self.lock:
            self.reads.append((session, bank_index, tuple(local), tuple(remote), tuple(lengths)))
            ticket = len(self.reads)
            self.active_reads += 1
            self.peak_reads = max(self.peak_reads, self.active_reads)
        try:
            if self.barrier is not None and ticket <= self.barrier.parties:
                self.barrier.wait(timeout=5)
            result = self.read_hook(bank_index) if self.read_hook is not None else None
            if result is not None:
                return result
            for dst, src, size in zip(local, remote, lengths):
                ctypes.memmove(dst, src, size)
            return 0
        finally:
            with self.lock:
                self.active_reads -= 1

    def assert_drained(self):
        assert not self.bank_owners and self.active_reads == 0
        assert all(not queue for queue in self.queues)
        for bank in self.reader._dsa_pd_mooncake_staging_banks:
            assert not any(bank.pending_slot_events)
            assert not bank.inflight_owners and not bank.needs_stream_drain
        assert all(not thread.is_alive() for thread in self.threads)
        assert not self.reader._dsa_pd_mooncake_staging_lock.locked()


def build(monkeypatch, workers, count=37, metadata=True):
    monkeypatch.setitem(envs.env_variables, "VLLM_ASCEND_DSA_PD_MOONCAKE_BANK_WORKERS", lambda: workers)
    monkeypatch.setattr(envs, "VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE", False)
    storage = torch.empty(ALIGNMENT + count * PIECE_BYTES + 64, dtype=torch.uint8)
    offset = (-storage.data_ptr()) % ALIGNMENT + 64
    source = storage[offset : offset + count * PIECE_BYTES]
    source.copy_((torch.arange(source.numel()) % 251).to(torch.uint8))
    host = torch.zeros(count * HOST_STRIDE, dtype=torch.uint8)
    registry.register_dsa_pd_mooncake_offloaded_kv_tensor(host)
    reader = make_reader(MagicMock(), pipeline=True, window=WINDOW)
    reader._pipeline_banks = BANKS
    batches = (count + WINDOW // BANKS - 1) // (WINDOW // BANKS)
    runtime = BankRuntime(monkeypatch, reader, min(workers, BANKS), batches)
    kwargs = {}
    if metadata:
        table = registry.create_dsa_pd_mooncake_transfer_targets([[host]])
        kwargs["transfer_targets"] = [table[0][0]] * count
    args = (
        "peer",
        [host.data_ptr() + index * HOST_STRIDE + 16 for index in range(count)],
        [source.data_ptr() + index * PIECE_BYTES for index in range(count)],
        [PIECE_BYTES] * count,
        "request",
    )
    return SimpleNamespace(
        reader=reader, runtime=runtime, source=source, host=host, args=args, kwargs=kwargs, count=count
    )


@pytest.mark.parametrize("workers", [0, 1, 2, 4, 7])
@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("count", [1, 37, 64])
def test_worker_counts_preserve_bytes_geometry_and_exclusive_bank_ownership(monkeypatch, workers, metadata, count):
    case = build(monkeypatch, workers, count, metadata)
    assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
    for index in range(count):
        assert torch.equal(
            case.host[index * HOST_STRIDE + 16 : index * HOST_STRIDE + 16 + PIECE_BYTES],
            case.source[index * PIECE_BYTES : (index + 1) * PIECE_BYTES],
        )
        assert torch.count_nonzero(case.host[index * HOST_STRIDE : index * HOST_STRIDE + 16]) == 0
        assert torch.count_nonzero(case.host[index * HOST_STRIDE + 16 + PIECE_BYTES : (index + 1) * HOST_STRIDE]) == 0
    reads = case.runtime.reads
    assert sum(sum(row[4]) for row in reads) == count * PIECE_BYTES
    assert len(reads) == (count + WINDOW // BANKS - 1) // (WINDOW // BANKS)
    for bank_index in range(BANKS):
        actual = [pointer for row in reads if row[1] == bank_index for pointer in row[3]]
        expected = [case.args[2][i] for i in range(count) if (i // (WINDOW // BANKS)) % BANKS == bank_index]
        assert actual == expected
    if workers <= 1:
        assert [pointer for row in reads for pointer in row[3]] == case.args[2]
    assert case.runtime.peak_reads == (min(workers, BANKS, len(reads)) if workers else 1)
    assert len(case.runtime.threads) <= min(workers, BANKS)
    assert case.reader._dsa_pd_mooncake_staging_bytes == WINDOW * ALIGNMENT
    assert sum(bank.views.capacity for bank in case.reader._dsa_pd_mooncake_staging_banks) == 256
    case.runtime.assert_drained()


@pytest.mark.parametrize("failure", ["negative", "read_exception", "copy_exception", "event_record"])
@pytest.mark.parametrize("workers", [2, 4])
def test_failure_waits_for_other_bank_and_drains_before_unlock(monkeypatch, failure, workers):
    case = build(monkeypatch, workers)
    peer_started, release_peer, failed = threading.Event(), threading.Event(), threading.Event()

    def read(bank):
        if bank == 1:
            peer_started.set()
            assert release_peer.wait(5)
        if bank == 0 and failure in ("negative", "read_exception"):
            assert peer_started.wait(5)
            failed.set()
            if failure == "negative":
                return -17
            raise RuntimeError("read failed")
        return None

    def fail(bank):
        if bank == 0:
            assert peer_started.wait(5)
            failed.set()
            raise RuntimeError("landing failed")

    case.runtime.read_hook = read
    if failure == "copy_exception":
        case.runtime.copy_hook = fail
    if failure == "event_record":
        case.runtime.event_hook = fail
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(case.reader.batch_transfer_sync_read, *case.args, **case.kwargs)
        try:
            assert failed.wait(5)
            assert not future.done()
            assert case.reader._dsa_pd_mooncake_staging_lock.locked()
        finally:
            release_peer.set()
        if failure == "negative":
            assert future.result(timeout=5) == -17
        else:
            with pytest.raises(RuntimeError, match="failed"):
                future.result(timeout=5)
    assert sum(row[1] == 0 for row in case.runtime.reads) == 1
    case.runtime.assert_drained()


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_repeated_requests_reuse_views_with_fresh_payloads(monkeypatch, workers):
    case = build(monkeypatch, workers)
    contexts = None
    for value in range(1, 7):
        case.source.fill_(value)
        case.host.zero_()
        assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
        if contexts is None:
            contexts = case.reader._dsa_pd_mooncake_staging_banks
        assert case.reader._dsa_pd_mooncake_staging_banks is contexts
        for index in range(case.count):
            assert torch.all(case.host[index * HOST_STRIDE + 16 : index * HOST_STRIDE + 16 + PIECE_BYTES] == value)
        assert sum(len(bank.views._views) for bank in contexts) <= 256
        case.runtime.assert_drained()


@pytest.mark.parametrize("failure", ["negative", "read_exception", "copy_exception", "event_record"])
def test_single_worker_failure_cancels_queued_bank_tasks(monkeypatch, failure):
    case = build(monkeypatch, 1)

    def read(_bank):
        if failure == "negative":
            return -17
        if failure == "read_exception":
            raise RuntimeError("read failed")
        return None

    def fail(_bank):
        raise RuntimeError("landing failed")

    case.runtime.read_hook = read
    if failure == "copy_exception":
        case.runtime.copy_hook = fail
    elif failure == "event_record":
        case.runtime.event_hook = fail
    if failure == "negative":
        assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == -17
    else:
        with pytest.raises(RuntimeError, match="failed"):
            case.reader.batch_transfer_sync_read(*case.args, **case.kwargs)
    assert len(case.runtime.reads) == 1
    case.runtime.assert_drained()


def test_second_request_cannot_enter_until_failed_request_has_joined(monkeypatch):
    case = build(monkeypatch, 2)
    peer_started, release_peer, failed, second_started = [threading.Event() for _ in range(4)]
    fail_once = True

    def read(bank):
        nonlocal fail_once
        if bank == 1 and not release_peer.is_set():
            peer_started.set()
            assert release_peer.wait(5)
        if bank == 0 and fail_once:
            fail_once = False
            assert peer_started.wait(5)
            failed.set()
            return -17
        return None

    case.runtime.read_hook = read

    def second():
        second_started.set()
        return case.reader.batch_transfer_sync_read("second", *case.args[1:], **case.kwargs)

    with ThreadPoolExecutor(max_workers=2) as callers:
        first = callers.submit(case.reader.batch_transfer_sync_read, *case.args, **case.kwargs)
        try:
            assert failed.wait(5)
            other = callers.submit(second)
            assert second_started.wait(5)
            assert not other.done() and case.reader._dsa_pd_mooncake_staging_lock.locked()
            assert all(row[0] != "second" for row in case.runtime.reads)
        finally:
            release_peer.set()
        assert first.result(timeout=5) == -17
        assert other.result(timeout=5) == 0
    case.runtime.assert_drained()


@pytest.mark.parametrize("reason", ["overlap", "probe", "pipeline_off"])
def test_worker_setting_retains_serial_fallbacks(monkeypatch, reason):
    case = build(monkeypatch, 4)
    case.runtime.workers = 0
    case.runtime.barrier = None
    if reason == "overlap":
        plan = case.reader._prepare_dsa_pd_mooncake_bank_plan

        def overlapping(*args):
            result = plan(*args)
            return staging_module._DSAPDMooncakeStagingPlan(result.batches, True)

        case.reader._prepare_dsa_pd_mooncake_bank_plan = overlapping
    elif reason == "probe":
        monkeypatch.setattr(envs, "VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE", True)
    else:
        case.reader._pipeline_landing_enabled = False
        # The synchronous fallback does not enter a landing stream context.
        case.reader._copy_dsa_pd_mooncake_staging_to_offloaded_kv = case.runtime.copy
    assert case.reader.batch_transfer_sync_read(*case.args, **case.kwargs) == 0
    assert not case.runtime.threads and case.runtime.peak_reads == 1
    assert [pointer for row in case.runtime.reads for pointer in row[3]] == case.args[2]
    case.runtime.assert_drained()


def test_overlapping_destinations_wait_for_prior_stream_before_next_batch(monkeypatch):
    case = build(monkeypatch, 4)
    case.runtime.workers = 0
    case.runtime.barrier = None
    destinations = [case.host.data_ptr() + 16] * case.count
    args = (case.args[0], destinations, *case.args[2:])
    assert case.reader.batch_transfer_sync_read(*args, **case.kwargs) == 0
    assert not case.runtime.threads
    assert torch.equal(case.host[16 : 16 + PIECE_BYTES], case.source[-PIECE_BYTES:])
    assert torch.count_nonzero(case.host[:16]) == 0
    assert torch.count_nonzero(case.host[16 + PIECE_BYTES :]) == 0
    assert [row[1] for row in case.runtime.reads] == [index % BANKS for index in range(len(case.runtime.reads))]
    case.runtime.assert_drained()


def test_direct_only_read_does_not_create_workers_or_touch_staging(monkeypatch):
    monkeypatch.setitem(envs.env_variables, "VLLM_ASCEND_DSA_PD_MOONCAKE_BANK_WORKERS", lambda: 4)
    source, target = torch.arange(32, dtype=torch.uint8), torch.zeros(32, dtype=torch.uint8)
    engine = MagicMock()

    def read(_session, local, remote, sizes):
        for dst, src, size in zip(local, remote, sizes):
            ctypes.memmove(dst, src, size)
        return 0

    engine.batch_transfer_sync_read.side_effect = read
    reader = make_reader(engine, pipeline=True, window=WINDOW)
    reader._ensure_dsa_pd_mooncake_staging_tensor = MagicMock(side_effect=AssertionError("unexpected staging"))
    monkeypatch.setattr(torch.npu, "set_device", MagicMock(side_effect=AssertionError("unexpected worker")))
    assert reader.batch_transfer_sync_read("peer", [target.data_ptr()], [source.data_ptr()], [32], "direct") == 0
    assert torch.equal(source, target)
    engine.batch_transfer_sync_read.assert_called_once()


def test_submission_failure_joins_started_tasks_and_drains(monkeypatch):
    case = build(monkeypatch, 2)

    class BrokenPool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.submits = 0

        def submit(self, *args, **kwargs):
            self.submits += 1
            if self.submits == 3:
                raise RuntimeError("submit failed")
            return super().submit(*args, **kwargs)

    # Queue failure must also work when only one started task reaches READ.
    case.runtime.barrier = None
    monkeypatch.setattr(staging_module, "ThreadPoolExecutor", BrokenPool)
    with pytest.raises(RuntimeError, match="submit failed"):
        case.reader.batch_transfer_sync_read(*case.args, **case.kwargs)
    case.runtime.assert_drained()


def test_worker_device_initialization_failure_does_not_read(monkeypatch):
    case = build(monkeypatch, 1)
    monkeypatch.setattr(torch.npu, "set_device", MagicMock(side_effect=RuntimeError("device failed")))
    with pytest.raises(RuntimeError, match="initializer|thread pool"):
        case.reader.batch_transfer_sync_read(*case.args, **case.kwargs)
    assert not case.runtime.reads
    case.runtime.assert_drained()


@pytest.mark.parametrize("value,expected", [(None, 0), ("0", 0), ("1", 1), ("2", 2), ("4", 4), ("8", 8)])
def test_worker_count_uses_central_env(monkeypatch, value, expected):
    name = "VLLM_ASCEND_DSA_PD_MOONCAKE_BANK_WORKERS"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert envs.env_variables[name]() == expected


@pytest.mark.parametrize("value", ["-1", "invalid", "1.5"])
def test_invalid_worker_count_is_rejected(monkeypatch, value):
    monkeypatch.setenv("VLLM_ASCEND_DSA_PD_MOONCAKE_BANK_WORKERS", value)
    with pytest.raises(ValueError):
        envs.env_variables["VLLM_ASCEND_DSA_PD_MOONCAKE_BANK_WORKERS"]()
