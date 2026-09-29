"""Pipeline-parallel forward for SGLang's qwen4_exp (Qwen3.8-Flash-Next) language model.

SGLang 0.5.20 already partitions qwen4_exp's layers across PP ranks (make_layers / PPMissingLayer via the Qwen3_5
base), skips out-of-stage weights at load, and sizes the stage-to-stage buffer for hyper-connection models as one
`hidden_states` tensor of width hidden_size * hc_count (configs/qwen4_exp.py maps hc_mult -> hc_count). What it does
not do is honour that buffer in the model: `Qwen4ExpModel.forward` always embeds and always runs the final
hyper-connection mixer, and `Qwen4ExpVLModel.forward` drops `pp_proxy_tensors`. This module restores the Qwen3_5
contract for qwen4_exp:

  first rank   embed tokens (or take input embeds), run [start_layer, end_layer)
  later ranks  take pp_proxy_tensors["hidden_states"] ([T, hc_count * hidden]; the layers' own stream layout)
  not last     return PPProxyTensors({"hidden_states": ...})     (qwen4_exp layers return residual=None)
  last rank    commit PLE history, final hyper-connection mix, same outputs as the stock forward

The PLE (n-gram embedding) batch is prepared and committed only on a rank that owns a PLE layer, so the n-gram table
lives on that rank alone. With pp_size == 1 every branch reduces to the stock forward.

Env: EXL3_PP_QWEN4EXP=0 disables the patch.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)
# EXL3_PP_MEM_TRACE=N: every N forwards, log torch reserved/allocated next to the driver's free device memory, per
# stage (separates torch caching-allocator growth from memory the runtime allocates outside torch).
_MEM_TRACE = int(os.environ.get("EXL3_PP_MEM_TRACE", "0"))
_mem_calls = [0]
# EXL3_TORCH_MEM_FRACTION caps the torch caching allocator on THIS stage's device. The plugin's activate() applies it
# before SGLang binds each PP rank to its card, so it only ever capped device 0; uncapped, a later stage's cache grew to
# 32.8 GiB reserved on a 31.9 GiB card during a 200K prefill and xe spilled buffers to system memory (then faulted).
_CAP = os.environ.get("EXL3_TORCH_MEM_FRACTION")
_capped = set()


def _cap_this_device() -> None:
    dev = torch.xpu.current_device()
    if not _CAP or dev in _capped:
        return
    torch.xpu.set_per_process_memory_fraction(float(_CAP), dev)
    _capped.add(dev)
    logger.warning("exl3xpu: torch XPU allocator capped at %.3f of device %d (this PP stage)", float(_CAP), dev)


def _mem_trace(model, forward_batch) -> None:
    _mem_calls[0] += 1
    if _mem_calls[0] % _MEM_TRACE:
        return
    g = 2 ** 30
    free, total = torch.xpu.mem_get_info()
    seq = getattr(forward_batch, "seq_lens_cpu", None)
    ctx = int(seq.max()) if seq is not None and len(seq) else -1
    logger.info("exl3xpu pp mem: stage %d-%d ctx %d mode %s | torch reserved %.2f allocated %.2f GiB | device free "
                "%.2f of %.2f GiB | outside torch %.2f GiB", model.start_layer, model.end_layer, ctx,
                forward_batch.forward_mode.name, torch.xpu.memory_reserved() / g, torch.xpu.memory_allocated() / g,
                free / g, total / g, (total - free - torch.xpu.memory_reserved()) / g)


def install() -> bool:
    from sglang.srt.models import qwen4_exp as q
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

    if getattr(q.Qwen4ExpModel.forward, "_exl3_pp", False):
        return True

    def _owns_ple(self) -> bool:
        if not self.has_ple:
            return False
        return any(getattr(self.layers[i], "ple", None) is not None for i in range(self.start_layer, self.end_layer))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,
                inputs_embeds: Optional[torch.Tensor] = None, pp_proxy_tensors: Optional[Any] = None):
        if _CAP and not _capped and not torch.xpu.is_current_stream_capturing():
            _cap_this_device()
        first = self.pp_group.is_first_rank
        last = self.pp_group.is_last_rank
        if first:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        else:
            assert pp_proxy_tensors is not None, "qwen4_exp PP: a later stage needs the previous stage's hidden states"
            hidden_states = pp_proxy_tensors["hidden_states"]

        ple_batch = (
            q._prepare_ple_batch(input_ids, forward_batch, ngram_size=self.ple_ngram_size,
                                 ngram_eos_token_id=self.ple_ngram_eos_token_id)
            if _owns_ple(self) else None
        )
        residual = None
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if i + 1 < self.end_layer:
                next_ple = getattr(self.layers[i + 1], "ple", None)
                if next_ple is not None:
                    next_ple.start_prefetch(ple_batch, forward_batch)
            with q.get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                    ple_batch=ple_batch,
                    captured_last_layer_outputs=(
                        aux_hidden_states if getattr(layer, "_is_layer_to_capture", False) else None
                    ),
                )

        q._commit_ple_batch(ple_batch, forward_batch)
        if _MEM_TRACE and not torch.xpu.is_current_stream_capturing():
            _mem_trace(self, forward_batch)

        if not last:
            if residual is not None:
                raise NotImplementedError("qwen4_exp PP: a layer returned a separate residual; the stage buffer "
                                          "carries only the hyper-connection streams")
            return PPProxyTensors({"hidden_states": hidden_states})

        hc_hidden_states = hidden_states
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        if not forward_batch.forward_mode.is_idle():
            return hidden_states, hc_hidden_states
        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states

    forward._exl3_pp = True
    q.Qwen4ExpModel.forward = forward

    base_vl_forward = q.Qwen4ExpVLModel.forward

    @torch.no_grad()
    def vl_forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None,
                   input_deepstack_embeds=None):
        self.last_hc_hidden_states = None
        if input_ids is None:
            input_ids = forward_batch.input_ids
        model_output = q.Qwen4ExpModel.forward(self, input_ids=input_ids, positions=positions,
                                               forward_batch=forward_batch, inputs_embeds=input_embeds,
                                               pp_proxy_tensors=pp_proxy_tensors)
        if isinstance(model_output, PPProxyTensors):
            return model_output
        if isinstance(model_output, tuple):
            hidden_states, self.last_hc_hidden_states = model_output
            return hidden_states
        return model_output

    vl_forward._exl3_pp = True
    vl_forward._exl3_base = base_vl_forward
    q.Qwen4ExpVLModel.forward = vl_forward

    # ModelRunner sets support_pp from the top-level forward's signature; the stock wrapper is (*args, **kwargs), so
    # SGLang refuses PP before loading finishes. Same behaviour, explicit parameters.
    @torch.no_grad()
    def cg_forward(self, input_ids, positions, forward_batch, get_embedding: bool = False, pp_proxy_tensors=None):
        output = q.Qwen3VLForConditionalGeneration.forward(self, input_ids, positions, forward_batch,
                                                           get_embedding=get_embedding,
                                                           pp_proxy_tensors=pp_proxy_tensors)
        hc_hidden_states = self.model.last_hc_hidden_states
        if hc_hidden_states is not None and isinstance(output, q.LogitsProcessorOutput):
            output.hidden_states = hc_hidden_states
        return output

    cg_forward._exl3_pp = True
    q.Qwen4ExpForConditionalGeneration.forward = cg_forward
    logger.info("exl3xpu: qwen4_exp pipeline-parallel forward installed")
    return True
