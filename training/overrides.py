"""Per-run changes to the settings in ``config``: ``--set section.field=value``.

The config values are frozen dataclasses that the code imports when it is
loaded, often as default arguments, so reassigning ``config.BC`` at run time
reaches nothing. Instead ``effective_configs`` builds new config objects with
``dataclasses.replace``, and ``training.train`` passes them explicitly to
every model and trainer.

Values are parsed with the dataclass field's type: ``int``, ``float``,
``bool`` (true/false, yes/no, 1/0), ``str``, ``none`` for optional fields, and
tuples as comma-separated lists (``slm.lora_targets=q_proj,k_proj,v_proj``).
"""

from __future__ import annotations

import types
import typing
from dataclasses import fields, replace
from typing import Any

import config

# --set section -> the config instance it changes.
SECTIONS: dict[str, str] = {"slm": "SLM", "bc": "BC", "ppo": "PPO", "symbolic": "SYMBOLIC", "train": "TRAIN"}

# Sections that decide which episodes exist: changing them needs new demos.
_DEMO_SECTIONS = ("env", "eval_sets", "data")


class OverrideError(ValueError):
    """A ``--set`` item that cannot be applied."""


def parse_value(text: str, tp: Any) -> Any:
    """Parse ``text`` as a value of type ``tp`` (a resolved type hint).

    Raises:
        OverrideError: ``text`` is not a valid value of that type.
    """
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        options = typing.get_args(tp)
        if type(None) in options and text.strip().lower() == "none":
            return None
        for option in (o for o in options if o is not type(None)):
            try:
                return parse_value(text, option)
            except OverrideError:
                pass
        raise OverrideError(f"{text!r} is not a valid {_type_name(tp)}")
    if origin is tuple:
        args = typing.get_args(tp)
        item = args[0] if args else str
        parts = [p.strip() for p in text.split(",")] if text.strip() else []
        return tuple(parse_value(p, item) for p in parts)
    if tp is bool:
        lowered = text.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
        raise OverrideError(f"{text!r} is not a bool (use true or false)")
    if tp in (int, float):
        try:
            return tp(text)
        except ValueError:
            raise OverrideError(f"{text!r} is not a valid {tp.__name__}") from None
    if tp is str:
        return text
    raise OverrideError(f"fields of type {_type_name(tp)} cannot be set from the command line")


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")


def parse_overrides(items: list[str], fixed: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Parse ``section.field=value`` items into ``{section: {field: value}}``.

    Args:
        items: the ``--set`` arguments.
        fixed: ``{"section.field": reason}`` for fields the caller sets itself
            (e.g. ``slm.backbone``, which follows ``--stage``).

    Raises:
        OverrideError: bad syntax, an unknown or fixed section / field, a field
            set twice, or a value of the wrong type.
    """
    fixed = fixed or {}
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        key, sep, text = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot or not name:
            raise OverrideError(f"--set {item!r}: expected section.field=value, e.g. bc.epochs=40")
        if section in _DEMO_SECTIONS:
            raise OverrideError(
                f"--set {item!r}: {section!r} settings change which episodes the demos contain, so they cannot "
                "be changed per run; edit config.py and re-collect with `python -m training.tools.collect_demos`"
            )
        if section not in SECTIONS:
            raise OverrideError(f"--set {item!r}: unknown section {section!r} (one of {', '.join(SECTIONS)})")
        cls = type(getattr(config, SECTIONS[section]))
        hints = typing.get_type_hints(cls)
        names = [f.name for f in fields(cls)]
        if name not in names:
            raise OverrideError(f"--set {item!r}: {cls.__name__} has no field {name!r} (fields: {', '.join(names)})")
        if f"{section}.{name}" in fixed:
            raise OverrideError(f"--set {item!r}: {fixed[f'{section}.{name}']}")
        if name in out.get(section, {}):
            raise OverrideError(f"--set {item!r}: {section}.{name} is set twice")
        try:
            value = parse_value(text, hints[name])
        except OverrideError as e:
            raise OverrideError(f"--set {item!r}: {e} (field type: {_type_name(hints[name])})") from None
        out.setdefault(section, {})[name] = value
    return out


def effective_configs(overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``{section: config object}`` for every section in ``SECTIONS``: the
    instance from ``config.py`` with that section's overrides applied."""
    return {section: replace(getattr(config, attr), **overrides.get(section, {})) for section, attr in SECTIONS.items()}
