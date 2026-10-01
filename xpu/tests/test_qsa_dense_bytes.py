"""QSA union attention (orig mode) on XPU: peak, time and output identity across EXL3_QSA_DENSE_BYTES score budgets.

The score block (rows per GEMM) is sized from _DENSE_SCORE_BYTES at call time; the serving settings in docs/b70-flashnext-pp2.md use 128 MiB. This
compares 64 MiB and 32 MiB with 128 MiB: smaller budgets only change how many query rows go into one score GEMM.
Shapes and slot patterns as tests/test_qsa_union_mem.py (24 q heads, 2 KV heads, head_dim 256, fp8 cache, 2048 rows x
2048 slots; random and local patterns at 16K/64K/128K/200K). Warm-up + 3 reps; peak above inputs; median ms.
  python test_qsa_dense_bytes.py [--out FILE]
Exit 0 only if every smaller budget is bit-identical to 128 MiB in every case.
"""
import argparse, json, os, statistics, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from exl3xpu import qsa_xpu as qx

ap = argparse.ArgumentParser()
ap.add_argument("--out")
a = ap.parse_args()
dev = torch.device("xpu", 0)
R, HQ, HK, D, TOPK = 2048, 24, 2, 256, 2048
BUDGETS = (128 << 20, 64 << 20, 32 << 20)
g = torch.Generator(device="cpu").manual_seed(0)


def slots_random(ctx):
    return torch.randint(0, ctx, (R, TOPK), generator=g)


def slots_local(ctx):
    rows = []
    for i in range(R):
        pos = ctx - R + i
        recent = torch.arange(max(0, pos - 511), pos + 1)
        blocks = torch.randint(0, max(1, (pos - 512) // 16), (96,), generator=g) * 16
        far = (blocks[:, None] + torch.arange(16)[None, :]).reshape(-1)
        rows.append(torch.cat([recent, far])[:TOPK])
    return torch.stack(rows)


def run(budget, q, kc, vc, ts):
    qx._DENSE_SCORE_BYTES = budget
    qx.qsa_sparse_attention_union(q, kc, vc, ts, mode="orig")
    torch.xpu.synchronize()
    peaks, times = [], []
    for _ in range(3):
        torch.xpu.empty_cache(); torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats(dev)
        base = torch.xpu.memory_allocated(dev)
        t = time.perf_counter()
        o = qx.qsa_sparse_attention_union(q, kc, vc, ts, mode="orig")
        torch.xpu.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
        peaks.append((torch.xpu.max_memory_allocated(dev) - base) / 2**20)
    return o.cpu(), max(peaks), statistics.median(times)


results, ok = [], True
for ctx in (16384, 65536, 131072, 200000):
    kc = torch.randn(ctx, HK, D, generator=g).to(dev).to(torch.float8_e4m3fn)
    vc = torch.randn(ctx, HK, D, generator=g).to(dev).to(torch.float8_e4m3fn)
    q = torch.randn(R, HQ, D, generator=g).to(dev, torch.bfloat16)
    for pattern, fn in (("random", slots_random), ("local", slots_local)):
        ts = fn(ctx).to(dev)
        nu = int(torch.unique(ts[ts >= 0]).numel())
        ref = None
        row = dict(context=ctx, pattern=pattern, union_slots=nu)
        for b in BUDGETS:
            o, pk, tm = run(b, q, kc, vc, ts)
            entry = dict(peak_mib=round(pk), median_ms=round(tm, 1))
            if ref is None:
                ref = o
            else:
                entry.update(bit_identical=torch.equal(o, ref), max_abs_diff=float((o.float() - ref.float()).abs().max()))
                ok &= entry["bit_identical"]
            row[f"{b >> 20}MiB"] = entry
        print(json.dumps(row), flush=True)
        results.append(row)
    del kc, vc, q
    torch.xpu.empty_cache()
if a.out:
    json.dump(dict(results=results, all_identical=ok), open(a.out, "w"), indent=1)
sys.exit(0 if ok else 1)
