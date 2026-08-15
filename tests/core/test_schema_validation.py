"""Tests for tool-argument validation against the advertised JSON Schema."""

from __future__ import annotations

import pytest

from openjarvis.core.schema_validation import ValidationError, validate_arguments

SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "timeout": {"type": "integer"},
        "ratio": {"type": "number"},
        "force": {"type": "boolean"},
        "paths": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": "string", "enum": ["read", "write"]},
    },
    "required": ["command"],
}


class TestRequired:
    def test_accepts_complete_arguments(self) -> None:
        params = {"command": "ls", "timeout": 5}
        assert validate_arguments(SCHEMA, params) == params

    def test_rejects_missing_required(self) -> None:
        with pytest.raises(ValidationError, match="missing required argument"):
            validate_arguments(SCHEMA, {"timeout": 5})

    def test_explicit_none_counts_as_missing(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            validate_arguments(SCHEMA, {"command": None})

    def test_reports_every_problem_at_once(self) -> None:
        """One turn per fix is expensive; surface the whole list."""
        with pytest.raises(ValidationError) as excinfo:
            validate_arguments(SCHEMA, {"timeout": "soon", "mode": "delete"})
        message = str(excinfo.value)
        assert "missing required argument" in message
        assert "'timeout'" in message
        assert "'mode'" in message

    def test_names_the_tool_when_given(self) -> None:
        with pytest.raises(ValidationError, match="for tool 'shell_exec'"):
            validate_arguments(SCHEMA, {}, tool_name="shell_exec")


class TestCoercion:
    """Several tools already tolerated stringified scalars — keep them working."""

    def test_numeric_string_becomes_integer(self) -> None:
        assert (
            validate_arguments(SCHEMA, {"command": "ls", "timeout": "30"})["timeout"]
            == 30
        )

    def test_numeric_string_becomes_float(self) -> None:
        assert (
            validate_arguments(SCHEMA, {"command": "ls", "ratio": "0.5"})["ratio"]
            == 0.5
        )

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("False", False), ("yes", True), ("off", False)],
    )
    def test_boolean_strings(self, raw: str, expected: bool) -> None:
        out = validate_arguments(SCHEMA, {"command": "ls", "force": raw})
        assert out["force"] is expected

    def test_number_becomes_string(self) -> None:
        out = validate_arguments(SCHEMA, {"command": 42})
        assert out["command"] == "42"

    def test_uncoercible_string_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="should be integer"):
            validate_arguments(SCHEMA, {"command": "ls", "timeout": "soon"})

    def test_does_not_mutate_the_caller_dict(self) -> None:
        params = {"command": "ls", "timeout": "30"}
        validate_arguments(SCHEMA, params)
        assert params["timeout"] == "30"


class TestTypes:
    def test_bool_is_not_an_integer(self) -> None:
        """Python says True == 1; a count argument should still reject it."""
        with pytest.raises(ValidationError, match="should be integer"):
            validate_arguments(SCHEMA, {"command": "ls", "timeout": True})

    def test_array_item_types_are_checked(self) -> None:
        with pytest.raises(ValidationError, match="only string values"):
            validate_arguments(SCHEMA, {"command": "ls", "paths": ["a", 2, "c"]})

    def test_array_of_correct_items_passes(self) -> None:
        out = validate_arguments(SCHEMA, {"command": "ls", "paths": ["a", "b"]})
        assert out["paths"] == ["a", "b"]

    def test_enum_membership(self) -> None:
        assert validate_arguments(SCHEMA, {"command": "ls", "mode": "read"})
        with pytest.raises(ValidationError, match="must be one of"):
            validate_arguments(SCHEMA, {"command": "ls", "mode": "delete"})

    def test_union_type_accepts_either(self) -> None:
        schema = {
            "type": "object",
            "properties": {"value": {"type": ["string", "integer"]}},
        }
        assert validate_arguments(schema, {"value": "x"})["value"] == "x"
        assert validate_arguments(schema, {"value": 3})["value"] == 3


class TestPermissiveness:
    """A tool with no contract must keep working — validation is not a gate."""

    @pytest.mark.parametrize("schema", [None, {}, {"type": "object"}, "nonsense"])
    def test_absent_or_unusable_schema_passes_through(self, schema: object) -> None:
        params = {"anything": 1}
        assert validate_arguments(schema, params) == params  # type: ignore[arg-type]

    def test_unknown_keys_allowed_by_default(self) -> None:
        """JSON Schema allows extras unless additionalProperties is false."""
        out = validate_arguments(SCHEMA, {"command": "ls", "extra": "kept"})
        assert out["extra"] == "kept"

    def test_unknown_keys_rejected_when_schema_closes_the_set(self) -> None:
        closed = {**SCHEMA, "additionalProperties": False}
        with pytest.raises(ValidationError, match="unexpected argument"):
            validate_arguments(closed, {"command": "ls", "extra": "x"})

    def test_property_without_a_type_is_not_checked(self) -> None:
        schema = {"type": "object", "properties": {"payload": {}}}
        assert validate_arguments(schema, {"payload": object()})


class TestAgainstRealTools:
    """Guard against a schema in the tree that this validator would reject."""

    def test_every_registered_tool_accepts_its_own_required_args(self) -> None:
        from openjarvis.agents.tool_resolver import ensure_registries_populated
        from openjarvis.core.registry import ToolRegistry

        # The suite clears registries between tests; this is the repair hatch
        # the project already provides for exactly that.
        ensure_registries_populated()

        checked = 0
        for name in ToolRegistry.keys():
            try:
                spec = ToolRegistry.get(name)().spec
            except Exception:
                continue  # tool needs constructor deps; covered elsewhere
            schema = spec.parameters
            if not isinstance(schema, dict):
                continue
            props = schema.get("properties")
            if not isinstance(props, dict):
                continue
            sample = {
                key: _sample_for(props[key])
                for key in schema.get("required", [])
                if key in props
            }
            validate_arguments(schema, sample, tool_name=name)
            checked += 1
        assert checked > 5, "registry looked empty — the guard proved nothing"


def _sample_for(subschema: dict) -> object:
    declared = subschema.get("type")
    if isinstance(declared, list):
        declared = declared[0] if declared else "string"
    if choices := subschema.get("enum"):
        return choices[0]
    return {
        "string": "x",
        "integer": 1,
        "number": 1.0,
        "boolean": True,
        "array": [],
        "object": {},
    }.get(declared, "x")


class TestReloadSafety:
    """The validator must sit outside the tool-reload blast radius.

    ``agents.tool_resolver.ensure_registries_populated`` reloads every
    ``openjarvis.tools.*`` module when the registry empties, and it runs on a
    production path. ``importlib.reload`` rebinds module globals in place, so
    a reloaded validator would make the *old* function raise the *new*
    ``ValidationError`` — which ``tools._stubs`` (excluded from the reload)
    would no longer catch, letting the exception escape ToolExecutor.execute.
    """

    def test_module_is_not_under_openjarvis_tools(self) -> None:
        from openjarvis.core import schema_validation

        assert not schema_validation.__name__.startswith("openjarvis.tools.")

    def test_repopulating_registries_keeps_the_gate_working(self) -> None:
        from openjarvis.agents.tool_resolver import ensure_registries_populated
        from openjarvis.core.registry import ToolRegistry
        from openjarvis.core.types import ToolCall
        from openjarvis.tools._stubs import ToolExecutor

        ToolRegistry.clear()
        ensure_registries_populated()

        from openjarvis.tools.think import ThinkTool

        executor = ToolExecutor([ThinkTool()])
        result = executor.execute(ToolCall(id="1", name="think", arguments="{}"))

        # The point is that execute() *returned* instead of raising.
        assert result.success is False
        assert "missing required argument(s): thought" in result.content
