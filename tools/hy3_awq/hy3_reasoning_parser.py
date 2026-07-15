# SPDX-License-Identifier: Apache-2.0
"""Hy3 reasoning parser plugin for vLLM.

Hy3's chat template denotes reasoning with HYTK-suffixed think tokens —
``<think:opensource>`` ... ``</think:opensource>`` (single tokens 120029 /
120030 in the released tokenizer). The opening tag is emitted by the chat
template inside the prompt (same convention as DeepSeek R1), so generated
text looks like ``reasoning</think:opensource>answer``. Subclassing the R1
parser reuses its no-start-token streaming handling; only the token strings
differ.

Without this parser the whole trace (including the literal closing tag)
lands in ``content``, and OpenRouter's reasoning validation fails on it.
``fast_start.sh serve`` loads it automatically; for a manual serve add:

    vllm serve ... \
      --reasoning-parser hy3 \
      --reasoning-parser-plugin /path/to/hy3_reasoning_parser.py

Thinking is gated per-request via ``reasoning_effort`` (no_think/low/high;
the chat template defaults to no_think) — the openinference rproxy already
maps OpenRouter's ``reasoning.effort`` onto that field.
"""

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


@ReasoningParserManager.register_module("hy3")
class Hy3ReasoningParser(DeepSeekR1ReasoningParser):
    """DeepSeek-R1-style parser with Hy3's ``:opensource``-suffixed tokens."""

    @property
    def start_token(self) -> str:
        return "<think:opensource>"

    @property
    def end_token(self) -> str:
        return "</think:opensource>"
