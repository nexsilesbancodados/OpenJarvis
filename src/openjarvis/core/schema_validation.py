"""Validate tool-call arguments against the JSON Schema a tool advertises.

``ToolSpec.parameters`` is handed to the model as the contract for a tool, but
nothing enforced it: :meth:`ToolExecutor.execute` only ran ``json.loads`` and
then splatted the result into ``tool.execute(**params)``. A missing required
key surfaced as a raw ``TypeError`` string, and a wrong-typed value surfaced
wherever the tool happened to break — both of which read to the model as "the
tool is broken" rather than "fix your arguments".

The checks here are deliberately narrow, because this runs in front of every
tool call in the system and a false rejection is worse than a late failure:

* **Missing required properties** are always an error. This is unambiguous.
* **Unknown properties** are an error only when the schema sets
  ``additionalProperties: false``. Schemas that stay silent keep accepting
  extras, as JSON Schema specifies.
* **Type mismatches** are coerced when the intent is unmistakable (the string
  ``"30"`` for an integer, ``"true"`` for a boolean) and rejected otherwise.
  Coercion matters because several tools already relied on models sending
  numbers as strings; failing those would be a regression, not a fix.

Nothing here needs a JSON Schema library. The schemas in this repo use a small
subset — ``type``, ``properties``, ``required``, ``items``, ``enum`` — and
pulling in a validator dependency for that would cost more than it returns.

This lives in ``core`` rather than next to the tools it serves, and that is
load-bearing. ``agents.tool_resolver.ensure_registries_populated`` reloads
every ``openjarvis.tools.*`` module (except ``_stubs``) whenever the tool
registry comes back empty, and it runs on a production path, not only under
test. ``importlib.reload`` rebinds a module's globals in place, so after a
reload the *old* :func:`validate_arguments` object raises the *new*
``ValidationError`` class — while ``tools._stubs``, which is excluded from the
reload, still holds the old one. Its ``except ValidationError`` would stop
matching and the exception would escape ``ToolExecutor.execute``. Nothing
blanket-reloads ``openjarvis.core.*``, so keeping the module here removes the
hazard rather than documenting it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

__all__ = ["ValidationError", "validate_arguments"]


class ValidationError(Exception):
    """Raised when arguments cannot be reconciled with a tool's schema."""


# JSON Schema type name -> accepted Python types. ``bool`` is excluded from the
# numeric entries on purpose: in Python ``True`` is an ``int``, and silently
# accepting a boolean where a count belongs hides real model mistakes.
_TYPES: Dict[str, Tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}

_TRUE = frozenset({"true", "yes", "1", "on"})
_FALSE = frozenset({"false", "no", "0", "off"})


def _matches(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    expected = _TYPES.get(type_name)
    if expected is None:
        # Unknown or absent type keyword — nothing to assert.
        return True
    if type_name in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _coerce(value: Any, type_name: str) -> Tuple[bool, Any]:
    """Attempt an unambiguous conversion. Returns ``(ok, converted)``."""
    if (
        type_name == "string"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return True, str(value)
    if not isinstance(value, str):
        return False, value

    text = value.strip()
    if type_name == "integer":
        try:
            return True, int(text)
        except ValueError:
            return False, value
    if type_name == "number":
        try:
            return True, float(text)
        except ValueError:
            return False, value
    if type_name == "boolean":
        low = text.lower()
        if low in _TRUE:
            return True, True
        if low in _FALSE:
            return True, False
    return False, value


def _type_names(schema: Dict[str, Any]) -> List[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, list):
        return [t for t in declared if isinstance(t, str)]
    return []


def validate_arguments(
    schema: Dict[str, Any] | None,
    params: Dict[str, Any],
    *,
    tool_name: str = "",
) -> Dict[str, Any]:
    """Check ``params`` against ``schema`` and return the coerced arguments.

    Raises :class:`ValidationError` with every problem found, not just the
    first — a model correcting one argument at a time burns a turn per fix.

    A missing, non-dict, or non-object schema validates trivially: tools are
    free to declare no contract, and this must not become a reason for a tool
    to stop working.
    """
    if not isinstance(schema, dict) or not isinstance(params, dict):
        return params
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return params

    label = f" for tool '{tool_name}'" if tool_name else ""
    problems: List[str] = []
    result = dict(params)

    required = schema.get("required")
    if isinstance(required, list):
        missing = [
            key for key in required if isinstance(key, str) and result.get(key) is None
        ]
        if missing:
            problems.append(
                "missing required argument(s): " + ", ".join(sorted(missing))
            )

    if schema.get("additionalProperties") is False:
        unknown = sorted(set(result) - set(properties))
        if unknown:
            problems.append(
                "unexpected argument(s): "
                + ", ".join(unknown)
                + ". Accepted: "
                + ", ".join(sorted(properties))
            )

    for key, value in list(result.items()):
        subschema = properties.get(key)
        if not isinstance(subschema, dict) or value is None:
            continue

        names = _type_names(subschema)
        if names and not any(_matches(value, name) for name in names):
            converted: Any = value
            for name in names:
                ok, candidate = _coerce(value, name)
                if ok:
                    converted = candidate
                    break
            else:
                problems.append(
                    f"argument '{key}' should be {' or '.join(names)},"
                    f" got {type(value).__name__}"
                )
                continue
            result[key] = converted
            value = converted

        choices = subschema.get("enum")
        if isinstance(choices, list) and choices and value not in choices:
            problems.append(
                f"argument '{key}' must be one of "
                + ", ".join(repr(c) for c in choices)
                + f"; got {value!r}"
            )

        item_schema = subschema.get("items")
        if (
            isinstance(item_schema, dict)
            and isinstance(value, (list, tuple))
            and (item_names := _type_names(item_schema))
        ):
            bad = [
                i
                for i, item in enumerate(value)
                if not any(_matches(item, name) for name in item_names)
            ]
            if bad:
                problems.append(
                    f"argument '{key}' must contain only"
                    f" {' or '.join(item_names)} values"
                    f" (bad index: {', '.join(str(i) for i in bad)})"
                )

    if problems:
        raise ValidationError(f"Invalid arguments{label}: " + "; ".join(problems))
    return result
