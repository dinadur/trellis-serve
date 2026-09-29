"""SGLang 0.5.20 serving fix: required / named tool_choice with a structural-tag tool parser (qwen3_coder).

At request time `OpenAIServingChat` asks the parser for a structural constraint; when none is produced (here: XPU,
thinking on) it falls back to the JSON-schema constraint, so the model emits a JSON array
`[{"name": ..., "parameters": {...}}]`. At response time `_process_tool_calls` sees a detector that "owns the format"
(supports_structural_tag), finds no XML tool-call markers, and returns before its own JSON-array branch, so the call is
delivered as plain content with tool_calls=None. Measured on Qwen3.8-Flash-Next: tool_choice auto parses correctly;
required and named calls are lost although the JSON is exact.

Fix: when a required/named result has no tool calls and the text is JSON, re-run the stock method with the parser
switched off, which takes the stock JSON-array branch. Nothing else changes.
Env: EXL3_SGL_REQUIRED_TOOL_FIX=0 disables it.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install() -> bool:
    from sglang.srt.entrypoints.openai import serving_chat as sc

    cls = sc.OpenAIServingChat
    if getattr(cls._process_tool_calls, "_exl3_fix", False):
        return True
    orig = cls._process_tool_calls

    def _process_tool_calls(self, text, tools, finish_reason, tool_choice=None, history_tool_calls_cnt=0):
        res = orig(self, text, tools, finish_reason, tool_choice, history_tool_calls_cnt)
        is_required = tool_choice == "required" or isinstance(tool_choice, sc.ToolChoice)
        if (res[0] is None and is_required and self.tool_call_parser and isinstance(text, str)
                and text.lstrip()[:1] in ("[", "{")):
            saved = self.tool_call_parser
            self.tool_call_parser = None
            try:
                res2 = orig(self, text, tools, finish_reason, tool_choice, history_tool_calls_cnt)
            finally:
                self.tool_call_parser = saved
            if res2[0]:
                return res2
        return res

    _process_tool_calls._exl3_fix = True
    cls._process_tool_calls = _process_tool_calls
    logger.info("exl3xpu: SGLang required/named tool-call JSON fallback installed")
    return True
