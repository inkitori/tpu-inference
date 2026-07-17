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

Thinking is gated per-request via ``chat_template_kwargs.reasoning_effort``
(no_think/low/high; the chat template defaults to no_think). NOTE: the
openinference rproxy must map OpenRouter's ``reasoning.effort`` onto that
field per model (rproxy ``handle_reasoning`` in main.rs) — as of 2026-07-17
it has no arm for tencent/Hy3, so OpenRouter reasoning requests arrive as
no_think until that mapping is added.
"""

from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


@ReasoningParserManager.register_module("hy3")
class Hy3ReasoningParser(DeepSeekR1ReasoningParser):
    """DeepSeek-R1-style parser with Hy3's ``:opensource``-suffixed tokens.

    Unlike R1, Hy3's thinking is optional. With ``reasoning_effort`` low/high
    the chat template ends the prompt with ``<think:opensource>`` and the
    output is ``reasoning</think:opensource>answer`` — exactly the R1 shape.
    But in the default ``no_think`` mode the template closes the think block
    *inside the prompt* (``<think:opensource></think:opensource>``), so the
    generated text contains no think tokens at all. The R1 base parser treats
    "no end token seen" as "still thinking" and would classify the entire
    answer as reasoning, returning ``content: null`` for every no_think
    request. The server constructs this parser per request with the effective
    ``chat_template_kwargs``, so gate on the requested effort: when thinking
    wasn't requested (and the model emitted no think tokens), everything is
    content.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        effort = (kwargs.get("chat_template_kwargs") or {}).get(
            "reasoning_effort", "no_think"
        )
        self._thinking = effort in ("low", "high")

    @property
    def start_token(self) -> str:
        return "<think:opensource>"

    @property
    def end_token(self) -> str:
        return "</think:opensource>"

    def _output_has_think_tokens(self, text_or_ids) -> bool:
        if isinstance(text_or_ids, str):
            return self.start_token in text_or_ids or self.end_token in text_or_ids
        return self.start_token_id in text_or_ids or self.end_token_id in text_or_ids

    def extract_reasoning(self, model_output, request):
        # no_think and no tags in the output -> it's all content. If the model
        # emitted think tokens anyway, defer to the R1 logic.
        if not self._thinking and not self._output_has_think_tokens(model_output):
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        if not self._thinking and not self._output_has_think_tokens(
            current_token_ids
        ):
            return DeltaMessage(content=delta_text)
        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Gates tool parsing / structured output on generated ids: in no_think
        # mode there is no reasoning phase to wait out.
        if not self._thinking and not self._output_has_think_tokens(input_ids):
            return True
        return super().is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self._thinking and not self._output_has_think_tokens(input_ids):
            return input_ids
        return super().extract_content_ids(input_ids)
