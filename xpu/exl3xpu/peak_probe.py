"""Diagnostic (EXL3_PEAK_PROBE=layers|children): where does stage memory peak during prefill, by context length?

Hooks run only on eager EXTEND (prefill) forwards, never during graph capture (decode replays run no hooks).
  layers   : every *DecoderLayer module; key = layer class name
  children : the direct sub-modules of every *DecoderLayer; key = layer class name + "." + child attribute name
Pre-hook: reset the torch peak counter and note the allocated bytes; post-hook: transient = peak - base. Hooked modules
in one mode are never nested, so resets do not disturb each other. The context (max sequence length of the batch) and
the mode come from a pre-hook on the model itself. Every EXL3_PEAK_PROBE_EVERY (default 32) prefill forwards the
cumulative table is logged: one line per (key, 32K context bucket) with max transient MiB and count.
Do not combine with EXL3_PP_MEM_TRACE (both reset the peak counter).
"""
from __future__ import annotations

import collections
import logging
import os

import torch

logger = logging.getLogger(__name__)
MODE = os.environ.get("EXL3_PEAK_PROBE", "")
EVERY = int(os.environ.get("EXL3_PEAK_PROBE_EVERY", "32"))
_state = {"ctx": -1, "extend": False, "n": 0}
_max = collections.defaultdict(float)
_cnt = collections.Counter()
_MIB = 2 ** 20


def _batch_info(args, kwargs):
    fb = kwargs.get("forward_batch")
    if fb is None:
        for a in args:
            if type(a).__name__ == "ForwardBatch":
                fb = a
                break
    if fb is None:
        return -1, False
    seq = getattr(fb, "seq_lens_cpu", None)
    ctx = int(seq.max()) if seq is not None and len(seq) else -1
    return ctx, fb.forward_mode.name == "EXTEND"


def _active() -> bool:
    return _state["extend"] and not torch.xpu.is_current_stream_capturing()


def _model_pre(mod, args, kwargs):
    _state["ctx"], _state["extend"] = _batch_info(args, kwargs)


def _model_post(mod, args, kwargs, out):
    if not _active():
        return
    _state["n"] += 1
    if _state["n"] % EVERY == 0:
        rows = sorted(_max.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        logger.info("exl3xpu peak probe (%s) after %d prefill forwards on %s: %s", MODE, _state["n"],
                    torch.xpu.current_device(),
                    "; ".join(f"{k}@{b}K={v:.0f}MiB/{_cnt[(k, b)]}" for (k, b), v in rows))


def _pre(mod, args):
    if _active():
        torch.xpu.reset_peak_memory_stats()
        mod._exl3_base = torch.xpu.memory_allocated()


def _post(mod, args, out):
    if _active() and hasattr(mod, "_exl3_base"):
        t = (torch.xpu.max_memory_allocated() - mod._exl3_base) / _MIB
        key = (mod._exl3_key, max(_state["ctx"], 0) // 32768 * 32)
        _max[key] = max(_max[key], t)
        _cnt[key] += 1


def attach(model: torch.nn.Module) -> int:
    if MODE not in ("layers", "children"):
        return 0
    # the language trunk: the first module whose class name ends in "Model" and that owns the decoder layers
    target = next((m for _, m in model.named_modules()
                   if type(m).__name__.endswith("Model") and isinstance(getattr(m, "layers", None), torch.nn.ModuleList)),
                  None)
    if target is None:
        raise RuntimeError("no module with a .layers ModuleList found")
    target.register_forward_pre_hook(_model_pre, with_kwargs=True)
    target.register_forward_hook(_model_post, with_kwargs=True)
    n = 0
    for name, m in model.named_modules():
        if not type(m).__name__.endswith("DecoderLayer"):
            continue
        cls = type(m).__name__.replace("Qwen4Exp", "").replace("DecoderLayer", "")
        hooked = [(cls, m)] if MODE == "layers" else \
                 [(f"{cls}.{cn}", c) for cn, c in m.named_children()]
        for key, mm in hooked:
            mm._exl3_key = key
            mm.register_forward_pre_hook(_pre)
            mm.register_forward_hook(_post)
            n += 1
    logger.info("exl3xpu peak probe (%s): %d modules hooked on %s", MODE, n, type(target).__name__)
    return n


def install() -> bool:
    if MODE not in ("layers", "children"):
        return False
    from sglang.srt.model_executor import model_runner as mr
    orig = mr.ModelRunner.load_model

    def load_model(self, *a, **k):
        r = orig(self, *a, **k)
        try:
            attach(self.model)
        except Exception as e:  # pragma: no cover
            logger.warning("exl3xpu: peak probe not attached (%s)", e)
        return r

    mr.ModelRunner.load_model = load_model
    return True
