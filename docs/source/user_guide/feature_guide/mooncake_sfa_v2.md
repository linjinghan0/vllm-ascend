# Mooncake V2 with SFA KV offload

On a supported Atlas A3 sparse-attention deployment, the decode node can use
`MooncakeConnectorV2` with ModelRunner V2 while sharing the existing
`SparseKVOffloadManager` and LIM + Copy-SFA attention backend.

Use ModelRunner V2 on both nodes:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

The prefill node uses `MooncakeConnectorV2` with `kv_role="kv_producer"` and
device-resident KV. On the decode node, use the following configuration in
addition to the deployment's model, tensor parallel and serving options:

```bash
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":27300}' \
--additional-config '{"sparse_kv_offload_config":{"enabled":true,"topk_buffer_size":2048,"dram_size_per_dp_GB":16,"use_fused_overlap":false}}'
```

Choose the Host capacity and hot-row budget for the deployment. The hot-row
budget must satisfy the existing LIM + Copy-SFA configuration requirements.
Do not also enable `dsa_sparse_attention_config.enable_cpu_kv_store` or
`dsa_sparse_attention_config.tp_shared_host_kv`: the SFA manager owns the Host
allocation and shared views in this mode.

The V2 connector publishes persistent Host main KV and HBM indexer/scale
components. SFA device placeholders are not PD payload. Main blocks are
partitioned across decode TP ranks into one shared Host pool; the indexer
remains device-resident and replicated as configured.

The D worker also registers its existing rank-private HBM hot/tail buffers.
For each request, V2 appends direct producer-to-HBM READs: a short prefix is
placed densely in the reserved hot row; a long prefix's incomplete last block
is placed at `hot_tokens + (logical_block % 2) * block_size` in the existing
two-block circular tail. An aligned long prefix requires no tail READ. These
HBM reads are replicated on every D TP rank, independently of shared Host block
ownership. The valid prefix ends before the last prompt token replayed by V2.

The existing V2 batch READ and bounded staging pipeline still land Host main
KV through the SFA manager's local-copy API. Receive completion includes both
direct HBM reads and the final Host staging drain. After all TP workers complete
PD, the first real decode validates the reservation and prefix and does not
repeat Host-to-hot initialization. Complete local prefix hits with no remote
PD and prefix rollback retain the existing local recovery path.

This adaptation changes only the V2 path. The existing SFA connector, transfer
flow, allocation and attention operators are unchanged; no shared PD path is
extracted.

Request admission and stable hot-row reservations reuse the SFA slot allocator.
Finished requests release their reservations. A cancelled in-flight PD retains
its row until all D workers finish their reads, preventing late HBM writes from
corrupting a reused row. A failed receive cannot
publish a ready hot row. Subsequent decode uses the existing SFA metadata,
LIM, Copy-SFA, hot-cache management and current-token Host writeback.

This combination retains existing SFA offload restrictions, including BF16 main
KV, supported device-resident indexer formats, and no context/pipeline parallel
offload. It selects LIM + Copy-SFA; `use_fused_overlap=true` is rejected for this
V2 combination. SFA offload uses native preprocessing, so performance comparisons
with the legacy DSA Host path must account for the backend change, hot-buffer
memory and the direct HBM hot/tail PD reads.
