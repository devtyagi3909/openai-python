from __future__ import annotations

import re
import json
import hashlib
from typing import Any, List, Generic, TypeVar

import httpx2
from pydantic import BaseModel

from openai import OpenAI
from openai.lib._tools import pydantic_function_tool, pydantic_responses_function_tool
from openai.lib._pydantic import normalize_schema_name
from openai.lib._parsing._responses import type_to_text_format_param
from openai.lib._parsing._completions import type_to_response_format_param

T = TypeVar("T")


class Item(BaseModel):
    title: str


class Page(BaseModel, Generic[T]):
    items: List[T]


class Envelope(BaseModel, Generic[T]):
    value: T


DeepGeneric = Envelope[Envelope[Envelope[Envelope[Envelope[Envelope[Page[Item]]]]]]]


NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def test_normalize_schema_name_basic() -> None:
    assert normalize_schema_name("Item") == "Item"
    assert normalize_schema_name("Page[Item]") == "Page_Item_"
    assert normalize_schema_name("Envelope[Page[Item]]") == "Envelope_Page_Item__"
    assert normalize_schema_name("Dict[str, int]") == "Dict_str__int_"
    assert (
        normalize_schema_name("Special.Name:With-Dashes_And_Underscores") == "Special_Name_With-Dashes_And_Underscores"
    )


def test_normalize_schema_name_length_boundary() -> None:
    # Exactly 64 chars: unchanged if already valid characters
    name_64 = "A" * 64
    assert normalize_schema_name(name_64) == name_64
    assert len(normalize_schema_name(name_64)) == 64
    assert NAME_PATTERN.fullmatch(normalize_schema_name(name_64))

    # Exactly 65 chars: truncated to 55 chars + '_' + 8 hex chars = 64 chars
    name_65 = "A" * 65
    norm_65 = normalize_schema_name(name_65)
    assert len(norm_65) == 64
    assert norm_65.startswith("A" * 55 + "_")
    expected_hash = hashlib.sha256(name_65.encode("utf-8")).hexdigest()[:8]
    assert norm_65 == f"{'A' * 55}_{expected_hash}"
    assert NAME_PATTERN.fullmatch(norm_65)


def test_normalize_schema_name_collision_resistance() -> None:
    # Two names differing only past 55 characters should yield distinct normalized names
    prefix = "A" * 55
    name1 = prefix + "B" * 20
    name2 = prefix + "C" * 20
    norm1 = normalize_schema_name(name1)
    norm2 = normalize_schema_name(name2)
    assert norm1 != norm2
    assert len(norm1) == 64
    assert len(norm2) == 64
    assert NAME_PATTERN.fullmatch(norm1)
    assert NAME_PATTERN.fullmatch(norm2)


def test_normalize_schema_name_idempotent() -> None:
    deep_name = DeepGeneric.__name__
    assert len(deep_name) > 64
    norm = normalize_schema_name(deep_name)
    assert len(norm) == 64
    assert normalize_schema_name(norm) == norm
    assert NAME_PATTERN.fullmatch(norm)


def test_pydantic_function_tool_generics() -> None:
    # Generic model
    tool = pydantic_function_tool(Page[Item])
    tool_name = tool["function"]["name"]
    assert tool_name == "Page_Item_"
    assert NAME_PATTERN.fullmatch(tool_name)

    # Deeply nested generic model
    deep_tool = pydantic_function_tool(DeepGeneric)
    deep_name = deep_tool["function"]["name"]
    assert len(deep_name) == 64
    assert NAME_PATTERN.fullmatch(deep_name)

    # Explicit name override is preserved
    explicit_tool = pydantic_function_tool(Page[Item], name="ExplicitPageTool")
    assert explicit_tool["function"]["name"] == "ExplicitPageTool"


def test_pydantic_responses_function_tool_generics() -> None:
    tool = pydantic_responses_function_tool(Page[Item])
    assert tool["name"] == "Page_Item_"
    assert NAME_PATTERN.fullmatch(tool["name"])

    deep_tool = pydantic_responses_function_tool(DeepGeneric)
    assert len(deep_tool["name"]) == 64
    assert NAME_PATTERN.fullmatch(deep_tool["name"])


def test_type_to_response_format_param_generics() -> None:
    rf = type_to_response_format_param(Page[Item])
    assert rf["type"] == "json_schema"
    name = rf["json_schema"]["name"]
    assert name == "Page_Item_"
    assert NAME_PATTERN.fullmatch(name)

    rf_deep = type_to_response_format_param(DeepGeneric)
    name_deep = rf_deep["json_schema"]["name"]
    assert len(name_deep) == 64
    assert NAME_PATTERN.fullmatch(name_deep)


def test_type_to_text_format_param_generics() -> None:
    tf = type_to_text_format_param(Page[Item])
    assert tf["type"] == "json_schema"
    assert tf["name"] == "Page_Item_"
    assert NAME_PATTERN.fullmatch(tf["name"])

    tf_deep = type_to_text_format_param(DeepGeneric)
    assert len(tf_deep["name"]) == 64
    assert NAME_PATTERN.fullmatch(tf_deep["name"])


def test_chat_completions_parse_request_name() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"items": [{"title": "Item 1"}]}),
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with OpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    ) as client:
        # Standard generic model
        result = client.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            response_format=Page[Item],
        )
        name = captured["body"]["response_format"]["json_schema"]["name"]
        assert name == "Page_Item_"
        assert NAME_PATTERN.fullmatch(name)
        assert result.choices[0].message.parsed is not None
        assert result.choices[0].message.parsed.items[0].title == "Item 1"

        # Deep generic exceeding 64 chars
        def deep_handler(request: httpx2.Request) -> httpx2.Response:
            captured["deep_body"] = json.loads(request.content)
            return httpx2.Response(
                200,
                json={
                    "id": "chatcmpl-test-2",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "value": {
                                            "value": {
                                                "value": {
                                                    "value": {
                                                        "value": {
                                                            "value": {
                                                                "items": [{"title": "Deep Item"}],
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        client_deep = OpenAI(
            api_key="sk-test",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(deep_handler)),
        )
        client_deep.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            response_format=DeepGeneric,
        )
        deep_name = captured["deep_body"]["response_format"]["json_schema"]["name"]
        assert len(deep_name) == 64
        assert NAME_PATTERN.fullmatch(deep_name)


def test_responses_parse_request_name() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1740000000,
                "model": "gpt-4o",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"items": [{"title": "Item 1"}]}),
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    with OpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    ) as client:
        # Standard generic model
        result = client.responses.parse(
            model="gpt-4o",
            input="hello",
            text_format=Page[Item],
        )
        name = captured["body"]["text"]["format"]["name"]
        assert name == "Page_Item_"
        assert NAME_PATTERN.fullmatch(name)
        assert result.output[0].content[0].parsed is not None  # type: ignore[union-attr]
        assert result.output[0].content[0].parsed.items[0].title == "Item 1"  # type: ignore[union-attr]

        # Deep generic exceeding 64 chars
        def deep_handler(request: httpx2.Request) -> httpx2.Response:
            captured["deep_body"] = json.loads(request.content)
            return httpx2.Response(
                200,
                json={
                    "id": "resp_test_2",
                    "object": "response",
                    "created_at": 1740000000,
                    "model": "gpt-4o",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {
                                            "value": {
                                                "value": {
                                                    "value": {
                                                        "value": {
                                                            "value": {
                                                                "value": {
                                                                    "items": [{"title": "Deep Item"}],
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    ),
                                }
                            ],
                        }
                    ],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                },
            )

        client_deep = OpenAI(
            api_key="sk-test",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(deep_handler)),
        )
        client_deep.responses.parse(
            model="gpt-4o",
            input="hello",
            text_format=DeepGeneric,
        )
        deep_name = captured["deep_body"]["text"]["format"]["name"]
        assert len(deep_name) == 64
        assert NAME_PATTERN.fullmatch(deep_name)
