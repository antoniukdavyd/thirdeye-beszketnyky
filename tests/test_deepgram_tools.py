import json

from assist.channel.deepgram_agent import dispatch_function_calls
from assist.perception.scene_store import SceneStore
from assist.tools import ToolRegistry, deepgram_think_functions, parse_tool_arguments


class _FakeReg:
    def execute(self, name, args):
        return {"ok": True, "name": name, "args": args}


def test_dispatch_function_calls_builds_responses():
    calls = [
        type(
            "C",
            (),
            {
                "id": "1",
                "name": "find_object",
                "arguments": '{"label":"door"}',
                "client_side": True,
            },
        )()
    ]
    out = dispatch_function_calls(_FakeReg(), calls)
    assert len(out) == 1
    assert out[0]["id"] == "1"
    assert out[0]["name"] == "find_object"
    body = json.loads(out[0]["content"])
    assert body["ok"] is True
    assert body["args"]["label"] == "door"


def test_deepgram_functions_are_client_side():
    reg = ToolRegistry(SceneStore())
    fns = deepgram_think_functions(reg)
    names = {f["name"] for f in fns}
    assert names >= {"sense_snapshot", "measure_distances", "find_object", "describe_scene"}
    for f in fns:
        assert f.get("client_side") is True
        assert "endpoint" not in f
        assert "parameters" in f and f["parameters"].get("type") == "object"


def test_parse_tool_arguments_json_string():
    assert parse_tool_arguments('{"label":"door"}') == {"label": "door"}
    assert parse_tool_arguments({}) == {}
