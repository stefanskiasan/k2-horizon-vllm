"""vLLM parser plugin for K2-Horizon-MoVA (IFM chat format).

Registers:
  * reasoning parser  "ifm"  — splits the <ifm|think>…</ifm|think> channel out
    (the opening tag lives in the generation prompt, so the model output begins
    inside the think channel and only emits the closing </ifm|think>).
  * tool parser       "ifm"  — parses the default XML tool-call format:
        <ifm|tool_calls>
        <ifm|tool_call>FUNC_NAME
        <ifm|arg_key>KEY</ifm|arg_key>
        [<ifm|arg_type>TYPE</ifm|arg_type>]
        <ifm|arg_value>VALUE</ifm|arg_value>
        …
        </ifm|tool_call>
        </ifm|tool_calls>
    and the JSON variant <ifm|tool_call>{"name":…,"arguments":{…}}</ifm|tool_call>.
"""
import json
import re
from collections.abc import Sequence

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module("ifm")
class IFMReasoningParser(BaseThinkingReasoningParser):
    # K2-Horizon has three reasoning-effort tiers, each with its own think tokens.
    # The tier is selected per request via chat_template_kwargs["reasoning_effort"]
    # (high | medium | low); default high, matching the chat template.
    _EFFORT_TOKENS = {
        "high": ("<ifm|think>", "</ifm|think>"),
        "medium": ("<ifm|think_fast>", "</ifm|think_fast>"),
        "low": ("<ifm|think_faster>", "</ifm|think_faster>"),
    }

    def __init__(self, tokenizer, *args, **kwargs):
        effort = (kwargs.get("chat_template_kwargs") or {}).get("reasoning_effort", "high")
        if not isinstance(effort, str) or effort not in self._EFFORT_TOKENS:
            effort = "high"
        self._start_token, self._end_token = self._EFFORT_TOKENS[effort]
        super().__init__(tokenizer, *args, **kwargs)

    @property
    def start_token(self) -> str:
        return self._start_token

    @property
    def end_token(self) -> str:
        return self._end_token


# ---------------- tool parser ----------------
from vllm.entrypoints.openai.engine.protocol import (  # noqa: E402
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from vllm.tool_parsers.abstract_tool_parser import (  # noqa: E402
    ToolParser,
    ToolParserManager,
)

_TOOLS_BLOCK = re.compile(r"<ifm\|tool_calls>(.*?)</ifm\|tool_calls>", re.DOTALL)
_ONE_CALL = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.DOTALL)
_ARG = re.compile(
    r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
    r"(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?"
    r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
    re.DOTALL,
)


def _coerce(value: str, typ: str | None) -> object:
    v = value.strip()
    t = (typ or "").strip().lower()
    if t in ("string", "str"):
        return v
    if t in ("integer", "int"):
        try:
            return int(v)
        except ValueError:
            pass
    if t in ("number", "float"):
        try:
            return float(v)
        except ValueError:
            pass
    if t in ("boolean", "bool"):
        return v.lower() == "true"
    if t in ("array", "object", "list", "dict") or (v and v[0] in "[{"):
        try:
            return json.loads(v)
        except (ValueError, json.JSONDecodeError):
            return v
    # no type hint: try JSON literal (numbers/bools/null/array/object), else string
    try:
        return json.loads(v)
    except (ValueError, json.JSONDecodeError):
        return v


def _parse_one_call(raw: str) -> dict | None:
    raw = raw.strip()
    # JSON variant: <ifm|tool_call>{"name":…,"arguments":{…}}</ifm|tool_call>
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if "name" in obj:
                return {"name": obj["name"], "arguments": obj.get("arguments", {})}
        except (ValueError, json.JSONDecodeError):
            pass
    # XML variant: first line is the function name, then arg_key/arg_value pairs
    name = raw.split("\n", 1)[0].strip()
    if not name:
        return None
    args: dict = {}
    for key, typ, val in _ARG.findall(raw):
        args[key.strip()] = _coerce(val, typ)
    return {"name": name, "arguments": args}


def _extract_calls(text: str) -> list[dict]:
    calls: list[dict] = []
    for block in _TOOLS_BLOCK.findall(text):
        for raw in _ONE_CALL.findall(block):
            c = _parse_one_call(raw)
            if c:
                calls.append(c)
    return calls


@ToolParserManager.register_module("ifm")
class IFMToolParser(ToolParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._streamed = False  # emit tool calls once, at end of stream

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        if "<ifm|tool_calls>" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        calls = _extract_calls(model_output)
        if not calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        pre = model_output.split("<ifm|tool_calls>", 1)[0].strip()
        tool_calls = [
            ToolCall(
                type="function",
                function=FunctionCall(
                    name=c["name"], arguments=json.dumps(c["arguments"], ensure_ascii=False)
                ),
            )
            for c in calls
        ]
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=pre or None
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Buffer approach: stream plain content until the tool block opens; once
        # the closing </ifm|tool_calls> has arrived, emit all tool calls at once.
        if "<ifm|tool_calls>" not in current_text:
            return DeltaMessage(content=delta_text) if delta_text else None
        # inside/after the tool block — suppress raw markup from content
        if "</ifm|tool_calls>" not in current_text or self._streamed:
            return None
        self._streamed = True
        calls = _extract_calls(current_text)
        if not calls:
            return None
        deltas = [
            DeltaToolCall(
                index=i,
                type="function",
                id=f"call_{i}",
                function=DeltaFunctionCall(
                    name=c["name"],
                    arguments=json.dumps(c["arguments"], ensure_ascii=False),
                ),
            )
            for i, c in enumerate(calls)
        ]
        return DeltaMessage(tool_calls=deltas)
