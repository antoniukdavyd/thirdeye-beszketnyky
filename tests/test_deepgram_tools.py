from assist.perception.scene_store import SceneStore
from assist.tools import ToolRegistry, deepgram_think_functions, parse_tool_arguments


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
