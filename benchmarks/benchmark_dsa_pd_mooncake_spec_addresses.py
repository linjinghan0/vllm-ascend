# SPDX-License-Identifier: Apache-2.0
"""Compare complete spec address planning against a Git revision on CPU.

Uses the production Python functions with synthetic cache addresses and a
TP-shared Host ownership predicate. No serving stack, DMA or NPU is required.
"""

import argparse
import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from benchmark_dsa_pd_mooncake_addresses import measure_variants

HOST_LAYERS = 78
INDEXER_LAYERS = 21
TP_SIZE = 8


def load_planner(root, ref=None):
    def read(name):
        return (
            subprocess.check_output(
                ["git", "-c", f"safe.directory={root}", "show", f"{ref}:{name}"], cwd=root, text=True
            )
            if ref
            else (root / name).read_text()
        )

    def function(source, name):
        return next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)

    source = read("vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py")
    tree = ast.parse(source)
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MooncakePullRecvingThread"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_append_block_transfer_addresses", "_append_spec_transfer_addresses")
    ]
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets)
    ]
    namespace = {"np": np, "is_circular_kv_cache_spec": lambda spec: False}
    for name in ("MambaSpec", "MLAAttentionSpec", "SlidingWindowMLASpec", "AscendSFAIndexerCacheSpec"):
        namespace[name] = type(name, (), {})
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *constants,
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_build_block_transfer_addresses"
            ),
            function(read("vllm_ascend/kv_offload/dsa_pd_shared_host.py"), "partition_shared_block_groups"),
            function(
                read("vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/utils.py"), "group_concurrent_contiguous"
            ),
            ast.ClassDef(name="Planner", bases=[], keywords=[], body=methods, decorator_list=[]),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<spec-address-benchmark>", "exec"), namespace)
    return namespace


def make_case(namespace, blocks, requests, fragmented, targets):
    thread = namespace["Planner"]()
    host_layers, indexer_layers = HOST_LAYERS, INDEXER_LAYERS
    layer_count = host_layers + indexer_layers
    thread.layer_names = [f"layer.{layer}" for layer in range(layer_count)]
    thread.kv_cache_specs = [namespace["MLAAttentionSpec"](), namespace["AscendSFAIndexerCacheSpec"]()]
    thread.kv_caches_base_addr = [
        [(1 << 40) + layer * (1 << 30) + component * (1 << 28) for component in range(2 if layer < host_layers else 1)]
        for layer in range(layer_count)
    ]
    remote_bases = [[address + (1 << 43) for address in layer] for layer in thread.kv_caches_base_addr]
    thread.block_lens = [[131072, 16384] if layer < host_layers else [16384] for layer in range(layer_count)]
    thread.block_strides = thread.block_lens
    thread._transfer_targets = targets
    host_bases = {base for layer in thread.kv_caches_base_addr[:host_layers] for base in layer}
    pool = SimpleNamespace(rank=3, size=TP_SIZE, contains=lambda address: address in host_bases)
    namespace["get_dsa_pd_shared_host_pool"] = lambda: pool
    peer = SimpleNamespace(kv_caches_base_addr=remote_bases)
    remote = SimpleNamespace(
        metadata_by_pcp_rank={0: SimpleNamespace(metadata_by_tp_rank={0: peer})},
        block_lens=thread.block_lens,
        block_strides=thread.block_strides,
    )
    entries = [
        (
            f"request-{request}",
            [request * 1000 + block * (3 if fragmented else 1) for block in range(blocks)],
            [request * 1000 + 100 + block * (5 if fragmented else 1) for block in range(blocks)],
        )
        for request in range(requests)
    ]
    by_spec = [
        {(layer, layer): entries for layer in range(host_layers)},
        {(layer, layer): entries for layer in range(host_layers, layer_count)},
    ]

    def run():
        src, dst, lengths, transfer_targets = [], [], [], []
        for index, layers in enumerate(by_spec):
            thread._append_spec_transfer_addresses(
                index, 0, 0, 1, TP_SIZE, 1, layers, remote, src, dst, lengths, transfer_targets
            )
        return src, dst, lengths, transfer_targets

    return run


def partition_calls(namespace, operation):
    original = namespace["partition_shared_block_groups"]
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    namespace["partition_shared_block_groups"] = counted
    try:
        operation()
    finally:
        namespace["partition_shared_block_groups"] = original
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    variants = {"baseline": load_planner(root, args.baseline_ref), "candidate": load_planner(root)}
    targets = [
        [object() for _ in range(2 if layer < HOST_LAYERS else 1)] for layer in range(HOST_LAYERS + INDEXER_LAYERS)
    ]
    results = []
    for blocks in (32, 94):
        for requests in (1, 4):
            for fragmented in (False, True):
                operations = {
                    name: make_case(namespace, blocks, requests, fragmented, targets)
                    for name, namespace in variants.items()
                }
                expected = operations["baseline"]()
                assert operations["candidate"]() == expected
                calls = {name: partition_calls(variants[name], operation) for name, operation in operations.items()}
                assert calls == {"baseline": 2 * HOST_LAYERS * requests, "candidate": requests}, calls
                measured = measure_variants(operations, args.samples)
                results.append(
                    dict(
                        blocks=blocks,
                        requests=requests,
                        fragmented=fragmented,
                        descriptors=len(expected[0]),
                        partition_calls=calls,
                        timings=measured,
                        speedup=measured["baseline"]["median_us"] / measured["candidate"]["median_us"],
                    )
                )
    result = dict(
        baseline_ref=args.baseline_ref,
        outputs_equal=True,
        samples=args.samples,
        results=results,
        scope="CPU synthetic planning: 78 Host layers with two components, 21 HBM indexer layers, fresh output lists.",
        limitations="Excludes native binding conversion, network transfer, NPU execution and end-to-end latency.",
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
