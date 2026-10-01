# AGENTS.md — serving Qwen3.8-Flash-Next on two Intel Arc Pro B70s (branch `b70-flashnext-pp2`)

Instructions for a coding agent helping someone run this branch. The narrative, evidence and benchmarks are in
`docs/b70-flashnext-pp2.md`; this file is the procedure. Everything here was validated on one host (n = 1), so
expect to adapt device names and paths, and tell the user when their host differs from the one in the guide. Work through the steps in order and stop at the first check
that fails: report what you saw instead of improvising around it.

## Ground rules

- Ask before anything that needs root, reboots the machine, or stops a service the user did not mention.
- Never print or store credentials. The model download may need a Hugging Face login and acceptance of the model
  licence (`qwen-community-1.0`): the user does both.
- Do not change these to "fix" a problem; each one is load-bearing (see the guide for why):
  - `EXL3_TORCH_MEM_FRACTION=0.91` — raising it lets the allocator oversubscribe VRAM; `xe` then spills to system
    memory and the card faults. Lower it if anything, never above 0.92.
  - `EXL3_NVME_PUBLISH=store` — `atomic` faulted on the tested host (no PCIe AtomicOps); use `store` for this recipe.
  - `CCL_TOPO_P2P_ACCESS=0` — the cards cannot DMA to each other on client root ports; pipeline tensors go via host.
  - `--chunked-prefill-size 2048`, `--max-running-requests 2`, `--mem-fraction-static 0.85`.
- One server per pair of cards. Check nothing else is using them (`sudo fuser -v /dev/dri/*`, or the user's own
  service list) before starting.

## 1. Check the host

Run and read, do not change anything yet:

```
lspci -nn | grep -i -E "vga|display"           # two Arc Pro B70 (Battlemage) devices
ls -l /dev/dri/by-path/                        # card + render node per PCI address
lspci -vvv -s <B70 address> | grep -i atomic   # expect AtomicOpsCtl: ReqEn- on client platforms (fine with store)
free -g; df -h <model dir>                     # 64 GB RAM worked; ~85 GB model on NVMe (the n-gram table is read from it)
docker --version
```

Needs: Linux with the `xe` kernel driver bound to both cards, Docker, 64 GB RAM or more, the model on NVMe (the
n-gram tier does random 102-byte reads from it). Note each card's PCI address and its `cardN` / `renderDN` nodes in
PCI order; the serving step needs them.

## 2. Get the model and the code

```
MODEL_DIR=/absolute/path/on/nvme/Qwen3.8-Flash-Next-exl3     # about 85 GB; used by every later step
IMAGE=ghcr.io/0xsero/sglang-exl3-xpu-flashnext@sha256:bce2f9ad6fd07113f75e091559882e81d68b6aeffb10f37f1f417d0881889a10
# branch 3.05bpw_h5_ng5, pinned to the commit that was tested
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 69e33439ae950f17bcbe95c98f117d80f759ab6d --local-dir "$MODEL_DIR"
git clone -b b70-flashnext-pp2 https://github.com/dinadur/trellis-serve
```

Check the files are complete (about 85 GB) before continuing.

## 3. Build the extensions inside the image

The image ships the compiler; the `.so` files are not in git, and mounting the checkout hides the image's own copies.
Use the exact command in the guide's "Running it" section. Then check:

```
ls -la trellis-serve/xpu/exl3xpu/{_C_sgl.so,_moe_sgl.so,exl3xpu_ngram_nvme.so}
```

All three must exist and be non-empty. If the build prints `error`, stop and show it.

## 4. Serve

Use the `docker run` in the guide, filling in the two cards' `card`/`renderD` nodes and a `by-path` directory that
contains only those two cards (the guide shows how). The first start takes several minutes (weights load, kernels
JIT-compile, decode graphs are captured). Wait for `GET http://127.0.0.1:8270/v1/models` to answer.

Then confirm from `docker logs <container>` that the fixes are active:

```
EXL3 qsa_xpu installed
EXL3 qsa prefill row-chunk budget on (128 MiB)
EXL3 qsa union memory mode auto (per-head from 32768 slots)
EXL3 n-gram NVMe host USM ... publish=store
exl3xpu_ngram_nvme: publish via uncached ESIMD stores
exl3xpu: /models/fnx: 75751 EXL3 linears, by bits {3: 75277, 5: 472, 4: 2}
```

`publish=store` only echoes the setting; `publish via uncached ESIMD stores` comes from the extension itself, so an old
build (for example a stale `EXL3_NVME_LIB`) shows the first line without the second. A missing line means the checkout or an environment variable did not reach the container: stop and fix that first.

## 5. Verify

```
curl -s http://127.0.0.1:8270/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"flashnext","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":256}'
sudo journalctl -k --since "-15 min" | grep -i -E "CAT error|fault response|device.?lost|gpu hang|wedged"
```

The kernel check must print nothing. Optional speed check (needs `pip install llama-benchy`):

```
llama-benchy --base-url http://127.0.0.1:8270/v1 --model turboderp/Qwen3.8-Flash-Next-exl3 \
  --served-model-name flashnext --tokenizer "$MODEL_DIR" --pp 2048 --tg 128 --depth 0 8192 \
  --concurrency 1 --runs 3 --no-cache
```

Expect roughly 1,600–2,200 tokens/s prefill and ~29 tokens/s decode at these depths on two B70s; the guide has
the full tables. The first
request after a start is slower (kernels warming up): ignore it.

## If something goes wrong

| symptom | likely cause | do this |
|---|---|---|
| kernel log `Engine memory CAT error`, `AccessType: 2` | atomic publish path in use | check the `publish=store` log line and `EXL3_NVME_PUBLISH`; check `EXL3_NVME_LIB` is unset or points at the new build |
| `torch.OutOfMemoryError` during a long prefill | a memory fix is not active, or another process holds VRAM | check the three `EXL3 qsa` log lines; check nothing else is on the cards |
| `mem_to_ipc_handle` / oneCCL errors at start | peer-to-peer attempted | `CCL_TOPO_P2P_ACCESS=0`; this branch moves stage tensors through host memory |
| prefill slows chunk by chunk, then a fault | VRAM oversubscribed, `xe` spilling to system memory | lower `EXL3_TORCH_MEM_FRACTION`; never raise it |
| host swaps or the container is killed | RAM: n-gram row cache plus the rest of the system (the guide caps the container at 32 GB) | lower `SGLANG_EXL3_NGRAM_RAM_GB` (8 by default here) |
| tracebacks at start mentioning `torchcodec` / `libavutil.so`, followed by `Ignore import error` | SGLang skipping optional audio processors (no FFmpeg in the image) | harmless; every validated start shows them |
| device lost and the server will not restart | the card is wedged | a reboot is usually needed; ask the user first |

To get more detail: `EXL3_PP_MEM_TRACE=1` logs per-stage memory and peaks; `EXL3_PEAK_PROBE=layers` logs per-layer
prefill peaks. Report the log lines and the kernel log rather than retrying the same start repeatedly.

## Tests (need a B70)

Inside the image, the checkout mounted, one card passed in with a `by-path` directory listing only that card (as in
the guide), the stress test also needs the model:

```
mkdir -p test-out testpath
ln -sfn ../cardA testpath/pci-0000:AA:00.0-card && ln -sfn ../renderDA testpath/pci-0000:AA:00.0-render
docker run --rm --device /dev/dri/cardA --device /dev/dri/renderDA -v "$PWD/testpath:/dev/dri/by-path:ro" \
  -v "$PWD/trellis-serve/xpu:/opt/trellis-serve/xpu" -v "$MODEL_DIR:/models/fnx:ro" -v "$PWD/test-out:/out" \
  --entrypoint bash "$IMAGE" -c 'cd /opt/trellis-serve/xpu &&
    python3 tests/test_qsa_prefill_mem.py --out /out/prefill_mem.json &&
    python3 tests/test_qsa_union_auto.py --out /out/union_auto.json &&
    python3 tests/test_ngram_nvme_stress.py --model /models/fnx --seconds 120 --out /out/stress.json'
```

Each test exits 0 on success and writes its JSON verdict to `test-out/`; the stress test pins about 2 GB of RAM.

## Afterwards

Offer to collect the user's results (hardware, versions, the startup log lines, any kernel faults, benchmark numbers)
in the form the guide's "Results from other hosts" section asks for, so they can share them if they wish. Do not post
anything on their behalf without asking.
