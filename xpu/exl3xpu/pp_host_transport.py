"""Host-staged pipeline-parallel transport for SGLang on Intel XPU hosts without peer-to-peer.

SGLang's `GroupCoordinator.send_tensor_dict` / `recv_tensor_dict` send device tensors with torch.distributed
point-to-point on the device group (oneCCL/XCCL). On hosts where the two B70s sit under separate PCIe root ports the
kernel refuses peer DMA, and oneCCL's point-to-point path fails creating an IPC handle
(`ze_handle_manager.cpp: mem_to_ipc_handle: device_fd is invalid value`). The same functions already send CPU tensors on
the CPU (gloo) group. This module moves device tensors to host memory before the stock send and moves those (and only those) received
tensors back onto the receiver's current XPU, so no oneCCL point-to-point is used. Stage payloads are small for
Qwen3.8-Flash-Next (hidden_size * hc_count bf16 = 20 KB per token).

Env: EXL3_PP_HOST_STAGED=0 disables it.
"""
from __future__ import annotations

import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


_MOVED = "__exl3_host_staged__"
_TIMING = os.environ.get("EXL3_STEP_TIMING", "0") != "0"
_acc = {"send": [0, 0.0], "recv": [0, 0.0]}


def timing_report() -> str:
    """Mean wall time of host-staged sends/recvs since the last report (EXL3_STEP_TIMING); resets the counters."""
    out = "; ".join(f"pp_{k} n={c} {1000 * s / c:.2f} ms" for k, (c, s) in _acc.items() if c)
    for v in _acc.values():
        v[0], v[1] = 0, 0.0
    return out


def _to_host(tensor_dict: dict) -> dict:
    """Device tensors -> host copies; the list of moved keys travels with the (pickled) metadata."""
    out, moved = {}, []
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor) and not v.is_cpu:
            v = v.to("cpu")          # synchronous: the host copy is complete before gloo reads it
            moved.append(k)
        out[k] = v
    if moved:
        out[_MOVED] = moved
    return out


def install() -> bool:
    from sglang.srt.distributed import parallel_state as ps

    GC = ps.GroupCoordinator
    if getattr(GC.send_tensor_dict, "_exl3_host", False):
        return True
    orig_send, orig_recv = GC.send_tensor_dict, GC.recv_tensor_dict

    def send_tensor_dict(self, tensor_dict, dst=None, all_gather_group=None, async_send=False):
        t = time.perf_counter()
        if isinstance(tensor_dict, dict) and self.world_size > 1:
            tensor_dict = _to_host(tensor_dict)
        r = orig_send(self, tensor_dict, dst=dst, all_gather_group=all_gather_group, async_send=async_send)
        if _TIMING:
            _acc["send"][0] += 1; _acc["send"][1] += time.perf_counter() - t
        return r

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        t = time.perf_counter()
        got = orig_recv(self, src=src, all_gather_group=all_gather_group)
        if isinstance(got, dict) and _MOVED in got:
            moved = set(got.pop(_MOVED))
            dev = torch.device("xpu", torch.xpu.current_device())
            got = {k: (v.to(dev) if k in moved else v) for k, v in got.items()}
        if _TIMING:     # includes waiting for the sender
            _acc["recv"][0] += 1; _acc["recv"][1] += time.perf_counter() - t
        return got

    send_tensor_dict._exl3_host = True
    recv_tensor_dict._exl3_host = True
    GC.send_tensor_dict, GC.recv_tensor_dict = send_tensor_dict, recv_tensor_dict
    logger.info("exl3xpu: pipeline-parallel tensors staged through host memory (gloo), no oneCCL point-to-point")
    return True
