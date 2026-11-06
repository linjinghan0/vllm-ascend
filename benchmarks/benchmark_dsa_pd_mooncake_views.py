# SPDX-License-Identifier: Apache-2.0
"""CPU metadata-only comparison of original versus cached landing slices.

Actual copy_, READ and device events are excluded. Report first access, cache
admission, repeated and changing-target cases separately; these are not NPU
performance measurements.
"""

import argparse
import importlib.util
import json
import statistics
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import torch

from vllm_ascend.kv_offload.dsa_pd_mooncake_staging import DSAPDMooncakeStagingReader

ALIGNMENT = 2 * 1024 * 1024
LAYERS = 78


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="95e59fd")
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", args.baseline_ref + ":vllm_ascend/kv_offload/dsa_pd_mooncake_staging.py"], cwd=root, text=True
    )
    baseline_path = args.output.with_suffix(".baseline.py")
    baseline_path.write_text(source)
    spec = importlib.util.spec_from_file_location("_landing_view_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    sizes = [ALIGNMENT * 2 + 128 * 1024, 512 * 1024 + 16 * 1024] * LAYERS
    targets = [torch.empty(size + 64, dtype=torch.int8) for size in sizes]
    staging = torch.empty(16 * ALIGNMENT, dtype=torch.int8)
    offsets = [
        0,
        3 * ALIGNMENT,
        4 * ALIGNMENT,
        7 * ALIGNMENT,
        8 * ALIGNMENT,
        11 * ALIGNMENT,
        12 * ALIGNMENT,
        15 * ALIGNMENT,
    ]
    readers = {
        "baseline": baseline.DSAPDMooncakeStagingReader.__new__(baseline.DSAPDMooncakeStagingReader),
        "candidate": DSAPDMooncakeStagingReader.__new__(DSAPDMooncakeStagingReader),
    }

    def request(reader, destination_offset):
        for i, (target, size) in enumerate(zip(targets, sizes)):
            reader._copy_dsa_pd_mooncake_staging_to_offloaded_kv(
                target.data_ptr() + destination_offset,
                size,
                offsets[i % len(offsets)],
                staging=staging,
                target_bytes=target,
                target_offset=destination_offset,
            )

    results = {}
    with patch.object(torch.Tensor, "copy_", lambda self, _source, non_blocking=False: self):
        for case in ["cold", "second_access", "repeated", "changing_target_offsets"]:
            samples = {name: [] for name in readers}
            for iteration in range(args.samples + 3):
                order = list(readers) if iteration % 2 == 0 else list(reversed(readers))
                for name in order:
                    reader = readers[name]
                    if case in ("cold", "second_access") and hasattr(reader, "_dsa_pd_mooncake_landing_views"):
                        del reader._dsa_pd_mooncake_landing_views
                    if case == "second_access":
                        request(reader, 0)
                    offset = 16 * (iteration % 4) if case == "changing_target_offsets" else 0
                    start = time.perf_counter_ns()
                    request(reader, offset)
                    elapsed = (time.perf_counter_ns() - start) / 1000
                    if iteration >= 3:
                        samples[name].append(elapsed)
            results[case] = {
                name: {"median_us": statistics.median(values), "samples_us": values} for name, values in samples.items()
            }
            results[case]["speedup"] = results[case]["baseline"]["median_us"] / results[case]["candidate"]["median_us"]
    result = {
        "baseline_ref": args.baseline_ref,
        "copies_per_request": len(targets),
        "staging_source_patterns": len(offsets),
        "cache_capacity": 256,
        "samples_per_case": args.samples,
        "results": results,
        "scope": "CPU slice construction and helper dispatch only; copy_ is a no-op.",
        "limitations": "No NPU, READ, hardware synchronization or end-to-end inference validation.",
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
