# Qwen3.8-Flash-Next EXL3 on two Intel Arc Pro B70s

This branch serves `turboderp/Qwen3.8-Flash-Next-exl3` across **two Arc Pro B70s** with SGLang pipeline parallelism
(PP=2, 24 layers per card, every routed expert in VRAM, 262K context), and fixes three problems that appear on
consumer/workstation hosts. **Everything below was measured on one machine (n = 1).** The defaults are what worked
there; treat the numbers as a single data point, not a guarantee. Each result names the configuration it was measured
with, because the fixes landed one at a time. Results from other hosts are welcome (see the end).

**Model.** The `3.05bpw_h5_ng5` build (commit `69e33439`): labelled 3.05 bpw, with the routed experts at 3 bits, a
5-bit output head (`h5`) and the n-gram embedding table quantised at K=5 (`ng5`; 102 bytes per 160-value row,
32.6 GB, served from NVMe through a RAM row cache). 85 GB (79 GiB) of tensor files.

## The host this was tested on

- 2 × Intel Arc Pro B70 (32 GB, Xe2/Battlemage), each PCIe Gen5 x8 on its own CPU root port
- Intel Core Ultra 7 270K Plus (Arrow Lake), Z890 board, 64 GB RAM, NVMe for the n-gram table
- Linux, `xe` kernel driver, Level Zero
- Image `ghcr.io/0xsero/sglang-exl3-xpu-flashnext@sha256:bce2f9ad6fd07113f75e091559882e81d68b6aeffb10f37f1f417d0881889a10`
  (SGLang 0.5.20, `94602c9c2`), with this branch mounted over `/opt/trellis-serve/xpu`

Two properties of this kind of host matter:

1. **No peer-to-peer DMA between the cards.** The kernel refuses P2P between devices on different client root ports.
   oneCCL point-to-point then fails creating an IPC handle, so this branch moves pipeline tensors through host memory
   (gloo) instead. Stage payloads are small (hidden 2560 × 4 hyper-connection streams, bf16 ≈ 20 KB per token).
2. **No PCIe AtomicOps to system memory.** On both B70s `AtomicOpsCtl: ReqEn-`, and the Arrow Lake root ports advertise
   no AtomicOpsCap (`lspci -vvv`). GPU atomic read-modify-writes to host (USM) memory are therefore not a safe primitive.

## Fix 1 — n-gram NVMe tier: publish without atomics (card-0 CAT faults)

**Symptom.** The first pipeline stage (card 0) dies at start-up (decode graph capture): xe `Engine memory CAT error`,
`AccessType: 2` (atomic) at a host-range address, `Fault response: Unsuccessful -EINVAL`, device lost. One fault was
also seen in steady decode, before buffer addresses were logged, so that one is not attributed.

**Cause.** `nvme_publish` (the n-gram PLE NVMe tier, stage 0 only) wrote every hashed id and its control words into
`sycl::malloc_host` memory with system-scope `atomic_ref::exchange`. With the buffer addresses logged, all 8
instrumented faults (one diagnostic start and the 7 faulting starts of the A/B below) were atomic accesses at exactly
the base of that `req` buffer.

**Fix.** Write ids and control words with ESIMD LSC stores that bypass L1/L3, each thread ending with a system-scope
fence; `req_n` and `req_seq` are written by one thread in a later kernel. `EXL3_NVME_PUBLISH=store` (the default) or
`atomic` (the old path). The store path is the default on every host but has only been exercised on this one, which
has no AtomicOps.

**Evidence.** A pre-registered 8 + 8 server A/B on alternating fresh boots: the atomic path faulted in **7/8** starts,
the store path in **0/8** (one-sided Fisher p ≈ 0.0007), every store start passing budget, long-decode and soak
checks. Configuration: store publish only (upstream QSA chunking, union `orig`), concurrent soak at 16K contexts.
In the stress test the store path returned 3.8 billion ids with 0 mismatches and was not slower. Not established:
whether missing PCIe AtomicOps is the exact mechanism, or why the atomic path faults only intermittently (a tight
microbenchmark never reproduced it). Store-path ordering relies on the payload → control kernel dependency surviving
XPU graph capture and on an aligned 32-bit uncached store being indivisible.

## Fix 2 — QSA indexer prefill: budget the whole torch fallback (64K OOM)

**Symptom.** A 64K-token prefill beside a concurrent request runs a pipeline stage out of memory under a 0.91 torch
memory cap: `torch.OutOfMemoryError` in `torch_qsa_mqa_prefill`.

**Cause.** On XPU, SGLang's QSA indexer prefill uses its torch fallback (TileLang runs only on CUDA). Row chunks are
sized so the `[rows, keys]` fp32 logits fit 128 MiB, but the fallback holds `[rows, keys, heads]` fp32 twice (scores,
ReLU). A 2048-row chunk at 64K peaked at **1,152 MiB**.

**Fix.** `exl3xpu/qsa_prefill_mem.py` sizes row chunks against the whole fallback (`EXL3_QSA_PREFILL_BUDGET_BYTES`,
default 128 MiB; `EXL3_QSA_PREFILL_BUDGET=0` disables). The per-row computation is unchanged: **93 MiB** at 64K with
bit-identical logits and indices in all six test cases, and equal or lower time.

**Evidence and limit.** With this fix alone the 64K OOM did not recur, but the one complete 64K–200K server start
still peaked 28.22 GiB on the second stage (0.80 GiB under its cap), above the 28.0 GiB target set beforehand. Fix 3
addresses the remaining transient.

## Fix 3 — QSA union attention: per KV head for long contexts

**Symptom.** With fix 2, a second prefill transient still grows with context in the full-attention layers.

**Cause.** `qsa_sparse_attention_union` gathers bf16 K/V for every slot any row of a chunk selected (~2 KB per slot)
plus score-block temporaries; the union approaches the whole context. An exact rewrite of the gather did not lower
the peak, and a smaller score budget (`EXL3_QSA_DENSE_BYTES`) cannot lower it at 128K and beyond (the score block is
already at its 8-row floor; at 32 MiB it also changes outputs at 16K).

**Fix.** `EXL3_QSA_UNION_MEM=auto` runs the union one KV head at a time when it has ≥ 32K slots and keeps the original
path below that, so short prompts are bit-identical. Long contexts differ at bf16-rounding level (max abs 4.9e-4,
relative RMS ≤ 2.2e-5). The 200K union peak falls from 999 to 548 MiB at a 128 MiB score budget (1,320 to 710 MiB at
the 384 MiB default). An unknown mode falls back to `orig`, and an invalid chunk size or threshold to its default
value, each with an error.

**Evidence.** Two server starts with fixes 1–3 under 64K–200K prompts beside a concurrent decode: the second stage
peaked at 27.83 GiB, **1.19 GiB** under the cap, every needle exact in both. Long-context quality on RULER-style
synthetic tasks (multi-key and multi-value needles, variable tracking; 32K, 64K, 128K, 200K; 144 items per arm):

| task | 32K | 64K | 128K | 200K |
|---|---|---|---|---|
| multi-key needle (`orig` / `auto`, of 12) | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |
| multi-value needle | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |
| variable tracking | 7.8 / 7.2 | 9.6 / 9.2 | 10.4 / 10.2 | 9.4 / 9.0 |

`auto` 131.6 vs `orig` 133.2 of 144 (document-clustered 95% interval of the difference [−3.8, +0.2]): inside the
5-point non-inferiority margin set beforehand, on this synthetic bank only. Variable tracking was lower with `auto` at
every context, and 4 of 36 documents scored worse (3 better). Greedy decoding was not deterministic run to run on this
stack: re-running 72 items moved `auto` by +1.8 (67/72 identical) and `orig` by 0 (70/72 identical). `auto` is
opt-in; the default stays `orig`.

## Running it

`AGENTS.md` at the top of this branch walks a coding agent (or a person) through the same steps with checks after
each one and a troubleshooting table. The model is branch `3.05bpw_h5_ng5`, tested at commit
`69e33439ae950f17bcbe95c98f117d80f759ab6d`
(`hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 69e33439ae950f17bcbe95c98f117d80f759ab6d --local-dir
"$MODEL_DIR"`).

Build the extensions inside the image, the way the image builds its own (it ships `icpx` and oneDNN; the compiled
`.so` files are not in git, and mounting the checkout hides the image's builds). Build on the serving host:
`build_ngram_nvme.sh` uses `-march=native`.

```
MODEL_DIR=/absolute/path/on/nvme/Qwen3.8-Flash-Next-exl3   # used by the commands below
git clone -b b70-flashnext-pp2 https://github.com/dinadur/trellis-serve trellis-serve
IMAGE=ghcr.io/0xsero/sglang-exl3-xpu-flashnext@sha256:bce2f9ad6fd07113f75e091559882e81d68b6aeffb10f37f1f417d0881889a10
docker run --rm -v "$PWD/trellis-serve/xpu:/opt/trellis-serve/xpu" --entrypoint bash "$IMAGE" -c '
  cd /opt/trellis-serve/xpu && X=exl3xpu &&
  EXL3_DNNL_DIR="$(python3 -c "import sys; print(sys.prefix)")" EXL3_OUT=$X/_C_sgl.so bash scripts/build_ext.sh &&
  EXL3_MOE_OUT=$X/_moe_sgl.so bash scripts/build_moe.sh && bash scripts/build_ngram_nvme.sh'
```

The build takes about a minute and writes the libraries as root into your checkout. The image's `EXL3_LIB` and
`EXL3_MOE_LIB` point at the `_sgl` names; the n-gram loader uses
`exl3xpu/exl3xpu_ngram_nvme.so` (set `EXL3_NVME_OUT` when building and `EXL3_NVME_LIB` when serving to keep a second
build beside it).

Serving passes in only the two cards, with a `by-path` directory listing just them (find each card's PCI address and
nodes with `ls -l /dev/dri/by-path/`):

```
mkdir -p bypath && cd bypath
ln -sfn ../cardA pci-0000:AA:00.0-card && ln -sfn ../renderDA pci-0000:AA:00.0-render
ln -sfn ../cardB pci-0000:BB:00.0-card && ln -sfn ../renderDB pci-0000:BB:00.0-render
cd ..
docker run -d --network host --ipc private --shm-size 16g --memory 32g --memory-swap 32g \
  --device /dev/dri/cardA --device /dev/dri/renderDA --device /dev/dri/cardB --device /dev/dri/renderDB \
  -v "$PWD/bypath:/dev/dri/by-path:ro" \
  -v "$MODEL_DIR:/models/fnx:ro" -v "$PWD/trellis-serve/xpu:/opt/trellis-serve/xpu:ro" \
  -e HF_HUB_OFFLINE=1 -e ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 -e CCL_TOPO_P2P_ACCESS=0 \
  -e EXL3_MOE_DEVICE_ALL=1 -e EXL3_MOE_SLOTS=0 -e EXL3_MOE_CACHE=0 \
  -e SGLANG_EXL3_NGRAM_TIER=nvme -e EXL3_NGRAM_TIER=nvme -e SGLANG_EXL3_NGRAM_RAM_GB=8 \
  -e EXL3_NVME_PUBLISH=store -e EXL3_QSA_UNION_MEM=auto -e EXL3_QSA_DENSE_BYTES=134217728 -e EXL3_QSA_ROWS=64 \
  -e EXL3_TORCH_MEM_FRACTION=0.91 -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  --entrypoint python3 "$IMAGE" -m sglang.launch_server \
  --model-path /models/fnx --served-model-name flashnext --quantization exl3 --trust-remote-code --device xpu \
  --pp-size 2 --dtype bfloat16 \
  --disable-shared-experts-fusion --kv-cache-dtype fp8_e4m3 --context-length 262144 --max-total-tokens 270336 \
  --mem-fraction-static 0.85 --chunked-prefill-size 2048 --max-running-requests 2 --max-mamba-cache-size 8 \
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --host 127.0.0.1 --port 8270
```

`--memory 32g` keeps the n-gram row cache and the rest of the server from pushing a 64 GB host into swap. The
indexer budget (fix 2) is on by default. Do not raise `EXL3_TORCH_MEM_FRACTION` to buy headroom: without a
per-stage cap the caching allocator oversubscribed VRAM during long prefills and `xe` spilled buffers to system
memory, then faulted. Non-torch usage is ~2.5 GiB per stage, so 0.91–0.92 is the ceiling on a 31.9 GiB B70.

## Benchmarks (one host, 2026-10-01)

A system-level comparison on this machine: engines, quantisation, speculative decoding and card count all differ.

- **A**: Qwen3.8-27B GGUF Q4_K_M + MTP draft head (3 tokens), llama.cpp SYCL, one card, f16 KV, 106K context, 2 slots sharing one KV cache.
- **B**: Flash-Next EXL3 3.05 bpw, this branch with fixes 1–3, both cards, settings above.
- **C**: Qwen3.8-27B EXL3 4.00 bpw + MTP (3 tokens), vLLM XPU, one card, fp8 KV.

`llama-benchy` 0.3.5: a 2,048-token prompt after the given context, 128 generated tokens, `--no-cache`, 2 runs per
cell. Prefill and decode are per-request tokens/s; first token is in seconds and includes the context prefill. For
two requests sent together: each one's first-token time, and the busiest one-second window of streamed tokens across
both. At long contexts the second prefill stalls the first request, so that window shows a brief overlap, not
sustained throughput.

Row marks: (a) A with a larger f16 context (135K; 139K for two 64K requests); (b) A with q8_0 KV and a 262K context
(264K for two 128K requests); (c) warm repeat, 3 runs after a discarded warm-up, replacing cold first runs (Flash-Next
0K prefill had read 779 and 3,408).

**A: 27B Q4_K_M + MTP, llama.cpp, one card**

| context | prefill | decode | first token | 2 requests: first tokens | 2 requests: peak 1 s |
|---|---|---|---|---|---|
| 0 | 756 | 36.7 | 3.0 | 5.7 / 5.7 | 57.5 |
| 8K | 920 | 32.9 | 11.4 | 21.7 / 24.8 | 50.5 |
| 32K | 854 | 31.2 | 41.0 | 62.6 / 100.5 | 43.5 |
| 64K (a) | 761 | 27.7 | 89.1 | 105.5 / 215.8 | 38.5 |
| 128K (ab) | 652 | 23.2 | 204.3 | 316.0 / 735.8 | 26.5 |
| 192K (b) | 553 | 11.6 | 359.5 | does not fit ² | — |
| 240K (b) | 501 | 10.8 | 495.0 | does not fit ² | — |

**B: Flash-Next, SGLang PP=2, two cards**

| context | prefill | decode | first token | 2 requests: first tokens | 2 requests: peak 1 s |
|---|---|---|---|---|---|
| 0 (c) | 1,633 | 28.8 | 1.5 | 1.8 / 2.5 | 50.7 |
| 8K (c) | 2,184 | 29.1 | 4.9 | 5.4 / 9.6 | 51.0 |
| 32K | 1,789 | 29.2 | 20.4 | 20.2 / 38.9 | 50.0 |
| 64K | 1,378 | 28.9 | 49.9 | 50.5 / 99.8 | 51.0 |
| 128K | 925 | 28.1 | 144.8 | 147.8 / 292.5 | 49.0 |
| 192K | 687 | 27.6 | 290.1 | does not fit ¹ | — |
| 240K | 592 | 27.7 | 419.3 | does not fit ¹ | — |

**C: 27B EXL3 + MTP, vLLM, one card**

| context | prefill | decode | first token | 2 requests: first tokens | 2 requests: peak 1 s |
|---|---|---|---|---|---|
| 0 (c) | 2,965 | 24.6 | 0.8 | 1.3 / 1.6 | 46.0 |
| 8K (c) | 2,610 | 23.6 | 4.0 | 5.2 / 8.4 | 44.0 |
| 32K | 2,235 | 21.2 | 15.7 | 15.8 / 36.0 | 36.0 |
| 64K | 1,853 | 18.6 | 36.6 | 39.0 / 95.5 | 30.0 |
| 128K | 1,337 | 15.2 | 99.7 | 106.1 / 285.0 | 17.5 |
| 192K | 1,099 | 12.8 | 180.9 | does not fit ¹ | — |
| 240K | 932 | 11.5 | 266.1 | does not fit ¹ | — |

¹ Two requests of 192K or more exceed the KV pool each server was configured with (269,952 tokens for B, 310,472 for
C, sized for a 262K context and two requests); another configuration might fit them.
² Two requests of 192K or more need at least 12.9 GiB of q8_0 KV; the 262K q8_0 server (8.5 GiB of KV) already fills
the card.

Memory placement for A, from the driver's statistics: the f16 servers up to 139K stay in VRAM (27.5–29.7 GiB); f16
sized for one 192K request put 19.5 GiB in system memory, hence q8_0 from 192K; the q8_0 servers fill the card
(31.6–31.9 of 31.9 GiB, about 0.5 GiB spilled). q8_0 also decodes slower (16.8 against 23.2 tokens/s at 128K), so the
(b) rows understate f16. The deepest context tested is 240K (the models' window is 262,144 tokens).

In short: Flash-Next's decode stays at 27.6–29.2 tokens/s from 0 to 240K, while A falls from 36.7 to 23.2 by 128K and
C from 24.6 to 11.5 by 240K. A decodes fastest at short contexts (its MTP draft head). C prefills fastest at every
depth; Flash-Next prefills 1.8–2.4× faster than A up to 64K, 1.4× at 128K and 1.2× beyond.

**Instruction following, IFBench-300** (same prompts, scorer and 4,096-token reasoning budget, two at a time;
runs on different dates):

| | loose | strict | date |
|---|---|---|---|
| A | 228 | 219 | 2026-08-20 |
| B (fix 1 only) | 231 | 220 | 2026-09-30 |
| C | 226 | 220 | 2026-09-29 |

Two runs of one identical configuration on this bank differed on 46 prompts, so these differences are within
run-to-run churn; none of the three is shown to be better. Flash-Next's long-context quality is under fix 3.

## Diagnostics on this branch

- `EXL3_PP_MEM_TRACE=N`: per-stage torch reserved/allocated and the peak since the previous trace, every N forwards.
- `EXL3_PEAK_PROBE=layers|children`: per-module prefill peak above each module's starting allocation, by 32K context
  bucket — how fix 3's transient was located.
- Tests: `xpu/tests/test_ngram_nvme_stress.py` (with `validate_subset_ref.py`), `test_qsa_prefill_mem.py`,
  `test_qsa_union_mem.py`, `test_qsa_union_auto.py`, `test_qsa_dense_bytes.py`.

## Results from other hosts

If you run this on different hardware, please open an issue or a pull request on this fork with what you found,
working or not: the CPU and board (and whether `lspci -vvv` shows AtomicOps on the root ports), GPU count and model,
kernel and `xe` driver version, image digest, the settings you changed, the startup log lines listed in `AGENTS.md`,
any kernel-log faults, and `llama-benchy` numbers if you have them. A host with PCIe AtomicOps that runs the atomic
publish path cleanly, or one where the store path faults, would be especially useful.

## Known limits

- One host, one image. Pipeline-parallel reliability rests on 8 fault-free starts with fix 1 (one-sided 95% upper
  bound on the per-start fault probability: 31%) and 2 further starts with fixes 1–3.
- Long-context quality was checked on synthetic retrieval and tracking tasks only.
- The benchmarks and server starts above ran on the tested tree; this branch adds the review fixes (setting
  validation, the indexer budget's own install block, removal of an unused union mode). The branch itself was
  built with the recipe above and served once end to end: all startup lines present, a chat request, a tool call,
  and prefill/decode at 0, 8K and 64K within a few percent of the tables, with no kernel faults.
- The intermittency of the atomic-path fault is unexplained.
