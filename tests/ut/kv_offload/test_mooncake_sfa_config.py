# SPDX-License-Identifier: Apache-2.0
"""The V2/SFA combination has one Host owner and an explicit runner/backend."""

from types import SimpleNamespace

import pytest
import vllm.envs

from vllm_ascend import ascend_config
from vllm_ascend.ascend_config import SparseKVOffloadConfig


def config(dsa=None):
    return SimpleNamespace(
        additional_config={"dsa_sparse_attention_config": dsa or {}},
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeConnectorV2",
            is_kv_consumer=True,
            has_connector=lambda _name: False,
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        speculative_config=None,
    )


@pytest.fixture(autouse=True)
def v2_a3(monkeypatch):
    monkeypatch.setattr(ascend_config, "is_950", lambda: False)
    monkeypatch.setattr(vllm.envs, "VLLM_USE_V2_MODEL_RUNNER", True)


def test_v2_consumer_selects_existing_lim_copy_backend():
    resolved = SparseKVOffloadConfig.from_additional_config(config(), {"enabled": True})
    assert resolved.enabled and resolved.use_fused_copy_sfa


@pytest.mark.parametrize("legacy", [{"enable_cpu_kv_store": True}, {"tp_shared_host_kv": True}])
def test_rejects_two_host_pool_owners(legacy):
    with pytest.raises(ValueError, match="SparseKVOffloadManager"):
        SparseKVOffloadConfig.from_additional_config(config(legacy), {"enabled": True})


def test_rejects_model_runner_v1(monkeypatch):
    monkeypatch.setattr(vllm.envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    with pytest.raises(ValueError, match="ModelRunner V2"):
        SparseKVOffloadConfig.from_additional_config(config(), {"enabled": True})


def test_rejects_unadapted_overlap_backend():
    with pytest.raises(ValueError, match="LIM \\+ Copy-SFA"):
        SparseKVOffloadConfig.from_additional_config(config(), {"enabled": True, "use_fused_overlap": True})
