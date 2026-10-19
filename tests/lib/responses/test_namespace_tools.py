from __future__ import annotations

import json
from typing import Any, cast

import httpx2
import pytest
from pydantic import BaseModel

import openai
from openai.types.responses import ToolParam
from openai.lib._parsing._responses import get_input_tool_by_name

from .test_commentary_parsing import _response


class CrmLookup(BaseModel):
    customer_id: str


class CatalogLookup(BaseModel):
    sku: int


class BillingLookup(BaseModel):
    account_id: str


def test_get_input_tool_by_name_direct() -> None:
    crm_tool = openai.pydantic_responses_function_tool(CrmLookup, name="lookup")
    catalog_tool = openai.pydantic_responses_function_tool(CatalogLookup, name="lookup")
    billing_tool = openai.pydantic_responses_function_tool(BillingLookup, name="invoice")

    crm_ns: ToolParam = {
        "type": "namespace",
        "name": "crm",
        "description": "CRM tools",
        "tools": [cast(Any, crm_tool)],
    }
    billing_ns: ToolParam = {
        "type": "namespace",
        "name": "billing",
        "description": "Billing tools",
        "tools": [cast(Any, billing_tool)],
    }

    tools: list[ToolParam] = [crm_ns, billing_ns, cast(ToolParam, catalog_tool)]

    # 1. Matching namespace finds namespaced tool
    res = get_input_tool_by_name(input_tools=tools, name="lookup", namespace="crm")
    assert res is not None
    assert res == crm_tool

    # 2. Namespace tool exists but does not contain the named function: strictly scopes, returns None
    res = get_input_tool_by_name(input_tools=tools, name="invoice", namespace="crm")
    assert res is None

    # 3. No namespace given: matches top-level tool
    res = get_input_tool_by_name(input_tools=tools, name="lookup", namespace=None)
    assert res is not None
    assert res == catalog_tool

    # 4. Namespace provided but does not name any namespace tool (deferred tool case):
    # falls back to searching top-level functions
    deferred_tools: list[ToolParam] = [{"type": "tool_search"}, cast(ToolParam, catalog_tool)]
    res = get_input_tool_by_name(input_tools=deferred_tools, name="lookup", namespace="lookup")
    assert res is not None
    assert res == catalog_tool

    # 5. Nonexistent tool with or without namespace returns None
    assert get_input_tool_by_name(input_tools=tools, name="missing", namespace="crm") is None
    assert get_input_tool_by_name(input_tools=tools, name="missing", namespace=None) is None
    assert get_input_tool_by_name(input_tools=tools, name="missing", namespace="unknown_ns") is None

    # 6. Namespace contains both custom tool and function tool with same name:
    # returns the function tool, not the custom tool
    custom_lookup = {"type": "custom", "name": "lookup"}
    ns_with_custom: ToolParam = {
        "type": "namespace",
        "name": "mixed",
        "description": "Mixed tools",
        "tools": [cast(Any, custom_lookup), cast(Any, crm_tool)],
    }
    assert get_input_tool_by_name(input_tools=[ns_with_custom], name="lookup", namespace="mixed") == crm_tool

    # 7. Namespace contains only custom tool with that name:
    # returns None since it is not a function tool
    ns_custom_only: ToolParam = {
        "type": "namespace",
        "name": "custom_only",
        "description": "Custom only",
        "tools": [cast(Any, custom_lookup)],
    }
    assert get_input_tool_by_name(input_tools=[ns_custom_only], name="lookup", namespace="custom_only") is None

    # 8. Multiple namespace blocks with same namespace name:
    # traverses all matching blocks and finds tools defined across blocks
    split_ns_1: ToolParam = {
        "type": "namespace",
        "name": "split",
        "description": "Split NS 1",
        "tools": [cast(Any, crm_tool)],
    }
    split_ns_2: ToolParam = {
        "type": "namespace",
        "name": "split",
        "description": "Split NS 2",
        "tools": [cast(Any, billing_tool)],
    }
    split_tools = [split_ns_1, split_ns_2]
    assert get_input_tool_by_name(input_tools=split_tools, name="lookup", namespace="split") == crm_tool
    assert get_input_tool_by_name(input_tools=split_tools, name="invoice", namespace="split") == billing_tool
    assert get_input_tool_by_name(input_tools=split_tools, name="missing", namespace="split") is None

    # 9. Robustness against non-dict items and malformed tools attribute
    malformed_tools: list[ToolParam] = [
        cast(Any, None),
        cast(Any, "invalid"),
        cast(ToolParam, {"type": "namespace", "name": "bad", "tools": cast(Any, 123)}),
        cast(ToolParam, {"type": "namespace", "name": "bad2", "tools": cast(Any, "not_a_list")}),
        cast(ToolParam, {"type": "function"}),  # missing name
    ]
    assert get_input_tool_by_name(input_tools=malformed_tools, name="lookup", namespace="bad") is None
    assert get_input_tool_by_name(input_tools=malformed_tools, name="lookup", namespace=None) is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_namespace_tool_scoping_and_collision(asynchronous: bool, streaming: bool) -> None:
    crm_tool = openai.pydantic_responses_function_tool(CrmLookup, name="lookup")
    catalog_tool = openai.pydantic_responses_function_tool(CatalogLookup, name="lookup")

    crm_ns: ToolParam = {
        "type": "namespace",
        "name": "crm",
        "description": "CRM tools",
        "tools": [cast(Any, crm_tool)],
    }
    tools: list[ToolParam] = [crm_ns, cast(ToolParam, catalog_tool)]

    items = [
        {
            "id": "fc_1",
            "call_id": "call_1",
            "type": "function_call",
            "name": "lookup",
            "namespace": "crm",
            "arguments": json.dumps({"customer_id": "C-42"}),
            "status": "completed",
        },
        {
            "id": "fc_2",
            "call_id": "call_2",
            "type": "function_call",
            "name": "lookup",
            "namespace": None,
            "arguments": json.dumps({"sku": 101}),
            "status": "completed",
        },
    ]

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert "tools" in json.loads(request.content)
        if not streaming:
            return httpx2.Response(200, json=_response(items))
        events = [
            {"type": "response.created", "response": _response([])},
            {"type": "response.output_item.added", "output_index": 0, "item": items[0]},
            {"type": "response.output_item.done", "output_index": 0, "item": items[0]},
            {"type": "response.output_item.added", "output_index": 1, "item": items[1]},
            {"type": "response.output_item.done", "output_index": 1, "item": items[1]},
            {"type": "response.completed", "response": _response(items)},
        ]
        body = "".join(f"data: {json.dumps({**event, 'sequence_number': i})}\n\n" for i, event in enumerate(events))
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    transport = httpx2.MockTransport(respond)
    if asynchronous:
        async with openai.AsyncOpenAI(
            api_key="synthetic", http_client=httpx2.AsyncClient(transport=transport, trust_env=False)
        ) as client:
            if streaming:
                async with client.responses.stream(model="test-model", input="lookup", tools=tools) as stream:
                    response = await stream.get_final_response()
            else:
                response = await client.responses.parse(model="test-model", input="lookup", tools=tools)
    else:
        with openai.OpenAI(
            api_key="synthetic", http_client=httpx2.Client(transport=transport, trust_env=False)
        ) as sync_client:
            if streaming:
                with sync_client.responses.stream(model="test-model", input="lookup", tools=tools) as sync_stream:
                    response = sync_stream.get_final_response()
            else:
                response = sync_client.responses.parse(model="test-model", input="lookup", tools=tools)

    call_crm = response.output[0]
    assert call_crm.type == "function_call"
    assert isinstance(call_crm.parsed_arguments, CrmLookup)
    assert call_crm.parsed_arguments.customer_id == "C-42"

    call_top = response.output[1]
    assert call_top.type == "function_call"
    assert isinstance(call_top.parsed_arguments, CatalogLookup)
    assert call_top.parsed_arguments.sku == 101


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_namespace_only_tool_parsing(asynchronous: bool, streaming: bool) -> None:
    crm_tool = openai.pydantic_responses_function_tool(CrmLookup, name="lookup")
    crm_ns: ToolParam = {
        "type": "namespace",
        "name": "crm",
        "description": "CRM tools",
        "tools": [cast(Any, crm_tool)],
    }
    tools: list[ToolParam] = [crm_ns]

    item = {
        "id": "fc_1",
        "call_id": "call_1",
        "type": "function_call",
        "name": "lookup",
        "namespace": "crm",
        "arguments": json.dumps({"customer_id": "C-99"}),
        "status": "completed",
    }

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert "tools" in json.loads(request.content)
        if not streaming:
            return httpx2.Response(200, json=_response([item]))
        events = [
            {"type": "response.created", "response": _response([])},
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": _response([item])},
        ]
        body = "".join(f"data: {json.dumps({**event, 'sequence_number': i})}\n\n" for i, event in enumerate(events))
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    transport = httpx2.MockTransport(respond)
    if asynchronous:
        async with openai.AsyncOpenAI(
            api_key="synthetic", http_client=httpx2.AsyncClient(transport=transport, trust_env=False)
        ) as client:
            if streaming:
                async with client.responses.stream(model="test-model", input="lookup", tools=tools) as stream:
                    response = await stream.get_final_response()
            else:
                response = await client.responses.parse(model="test-model", input="lookup", tools=tools)
    else:
        with openai.OpenAI(
            api_key="synthetic", http_client=httpx2.Client(transport=transport, trust_env=False)
        ) as sync_client:
            if streaming:
                with sync_client.responses.stream(model="test-model", input="lookup", tools=tools) as sync_stream:
                    response = sync_stream.get_final_response()
            else:
                response = sync_client.responses.parse(model="test-model", input="lookup", tools=tools)

    call = response.output[0]
    assert call.type == "function_call"
    assert isinstance(call.parsed_arguments, CrmLookup)
    assert call.parsed_arguments.customer_id == "C-99"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_deferred_top_level_fallback(asynchronous: bool, streaming: bool) -> None:
    deferred = openai.pydantic_responses_function_tool(CatalogLookup, name="lookup", defer_loading=True)
    tools: list[ToolParam] = [{"type": "tool_search"}, cast(ToolParam, deferred)]

    # Deferred tool call returned with namespace == name
    item = {
        "id": "fc_1",
        "call_id": "call_1",
        "type": "function_call",
        "name": "lookup",
        "namespace": "lookup",
        "arguments": json.dumps({"sku": 7}),
        "status": "completed",
    }

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert "tools" in json.loads(request.content)
        if not streaming:
            return httpx2.Response(200, json=_response([item]))
        events = [
            {"type": "response.created", "response": _response([])},
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": _response([item])},
        ]
        body = "".join(f"data: {json.dumps({**event, 'sequence_number': i})}\n\n" for i, event in enumerate(events))
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    transport = httpx2.MockTransport(respond)
    if asynchronous:
        async with openai.AsyncOpenAI(
            api_key="synthetic", http_client=httpx2.AsyncClient(transport=transport, trust_env=False)
        ) as client:
            if streaming:
                async with client.responses.stream(model="test-model", input="lookup", tools=tools) as stream:
                    response = await stream.get_final_response()
            else:
                response = await client.responses.parse(model="test-model", input="lookup", tools=tools)
    else:
        with openai.OpenAI(
            api_key="synthetic", http_client=httpx2.Client(transport=transport, trust_env=False)
        ) as sync_client:
            if streaming:
                with sync_client.responses.stream(model="test-model", input="lookup", tools=tools) as sync_stream:
                    response = sync_stream.get_final_response()
            else:
                response = sync_client.responses.parse(model="test-model", input="lookup", tools=tools)

    call = response.output[0]
    assert call.type == "function_call"
    assert isinstance(call.parsed_arguments, CatalogLookup)
    assert call.parsed_arguments.sku == 7


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_strict_plain_tool_in_namespace(asynchronous: bool, streaming: bool) -> None:
    strict_tool: ToolParam = {
        "type": "function",
        "name": "raw_lookup",
        "description": "raw strict function",
        "strict": True,
        "parameters": {"type": "object", "properties": {"code": {"type": "integer"}}, "required": ["code"]},
    }
    non_strict_tool: ToolParam = {
        "type": "function",
        "name": "loose_lookup",
        "description": "raw non-strict function",
        "strict": False,
        "parameters": {"type": "object"},
    }
    crm_ns: ToolParam = {
        "type": "namespace",
        "name": "crm",
        "description": "CRM tools",
        "tools": [cast(Any, strict_tool), cast(Any, non_strict_tool)],
    }
    tools: list[ToolParam] = [crm_ns]

    items = [
        {
            "id": "fc_1",
            "call_id": "call_1",
            "type": "function_call",
            "name": "raw_lookup",
            "namespace": "crm",
            "arguments": json.dumps({"code": 404}),
            "status": "completed",
        },
        {
            "id": "fc_2",
            "call_id": "call_2",
            "type": "function_call",
            "name": "loose_lookup",
            "namespace": "crm",
            "arguments": json.dumps({"extra": True}),
            "status": "completed",
        },
    ]

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert "tools" in json.loads(request.content)
        if not streaming:
            return httpx2.Response(200, json=_response(items))
        events = [
            {"type": "response.created", "response": _response([])},
            {"type": "response.output_item.added", "output_index": 0, "item": items[0]},
            {"type": "response.output_item.done", "output_index": 0, "item": items[0]},
            {"type": "response.output_item.added", "output_index": 1, "item": items[1]},
            {"type": "response.output_item.done", "output_index": 1, "item": items[1]},
            {"type": "response.completed", "response": _response(items)},
        ]
        body = "".join(f"data: {json.dumps({**event, 'sequence_number': i})}\n\n" for i, event in enumerate(events))
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    transport = httpx2.MockTransport(respond)
    if asynchronous:
        async with openai.AsyncOpenAI(
            api_key="synthetic", http_client=httpx2.AsyncClient(transport=transport, trust_env=False)
        ) as client:
            if streaming:
                async with client.responses.stream(model="test-model", input="lookup", tools=tools) as stream:
                    response = await stream.get_final_response()
            else:
                response = await client.responses.parse(model="test-model", input="lookup", tools=tools)
    else:
        with openai.OpenAI(
            api_key="synthetic", http_client=httpx2.Client(transport=transport, trust_env=False)
        ) as sync_client:
            if streaming:
                with sync_client.responses.stream(model="test-model", input="lookup", tools=tools) as sync_stream:
                    response = sync_stream.get_final_response()
            else:
                response = sync_client.responses.parse(model="test-model", input="lookup", tools=tools)

    call_strict = response.output[0]
    assert call_strict.parsed_arguments == {"code": 404}

    call_loose = response.output[1]
    assert call_loose.parsed_arguments is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_namespace_mixed_custom_and_function_tools(asynchronous: bool, streaming: bool) -> None:
    crm_func = openai.pydantic_responses_function_tool(CrmLookup, name="lookup")
    crm_custom: ToolParam = {
        "type": "custom",
        "name": "lookup",
        "description": "custom lookup",
    }
    crm_ns: ToolParam = {
        "type": "namespace",
        "name": "crm",
        "description": "CRM tools",
        "tools": [crm_custom, cast(Any, crm_func)],
    }
    tools: list[ToolParam] = [crm_ns]

    item = {
        "id": "fc_1",
        "call_id": "call_1",
        "type": "function_call",
        "name": "lookup",
        "namespace": "crm",
        "arguments": json.dumps({"customer_id": "C-123"}),
        "status": "completed",
    }

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert "tools" in json.loads(request.content)
        if not streaming:
            return httpx2.Response(200, json=_response([item]))
        events = [
            {"type": "response.created", "response": _response([])},
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": _response([item])},
        ]
        body = "".join(f"data: {json.dumps({**event, 'sequence_number': i})}\n\n" for i, event in enumerate(events))
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    transport = httpx2.MockTransport(respond)
    if asynchronous:
        async with openai.AsyncOpenAI(
            api_key="synthetic", http_client=httpx2.AsyncClient(transport=transport, trust_env=False)
        ) as client:
            if streaming:
                async with client.responses.stream(model="test-model", input="lookup", tools=tools) as stream:
                    response = await stream.get_final_response()
            else:
                response = await client.responses.parse(model="test-model", input="lookup", tools=tools)
    else:
        with openai.OpenAI(
            api_key="synthetic", http_client=httpx2.Client(transport=transport, trust_env=False)
        ) as sync_client:
            if streaming:
                with sync_client.responses.stream(model="test-model", input="lookup", tools=tools) as sync_stream:
                    response = sync_stream.get_final_response()
            else:
                response = sync_client.responses.parse(model="test-model", input="lookup", tools=tools)

    call = response.output[0]
    assert call.type == "function_call"
    assert isinstance(call.parsed_arguments, CrmLookup)
    assert call.parsed_arguments.customer_id == "C-123"
