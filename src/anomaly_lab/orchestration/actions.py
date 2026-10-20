"""actions: concrete operations that can be registered and executed.

No LLM. No autonomous loops.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class ActionResult:
    """Result of executing an Action."""

    action_name: str
    success: bool
    output: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


# Action signature: (context: dict[str, Any], **kwargs) -> ActionResult
SkillAction = Callable[..., ActionResult]


@dataclass(frozen=True, kw_only=True)
class Action:
    """A named, executable unit of work registered in the ActionRegistry.

    Unlike WorkflowStep (which is a protocol declaration), Action is the
    runtime executable unit.  The name must match what is registered.
    """

    name: str
    fn: SkillAction

    def execute(self, context: dict[str, Any], **kwargs: Any) -> ActionResult:
        return self.fn(context, **kwargs)


@dataclass
class ActionRegistry:
    """Mutable registry of named SkillActions.

    Actions are registered by name and retrieved for execution.
    The registry is a plain container — it does not execute anything itself.
    """

    _actions: dict[str, SkillAction] = field(default_factory=dict)

    def register(self, name: str, fn: SkillAction) -> None:
        """Register a skill action under ``name``."""
        self._actions[name] = fn

    def get(self, name: str) -> SkillAction | None:
        return self._actions.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._actions.keys()))


# Global registry instance — shared across the process
_default_registry = ActionRegistry()


def act(
    action_name: str,
    context: dict[str, Any],
    *,
    registry: ActionRegistry | None = None,
    **kwargs: Any,
) -> ActionResult:
    """Look up and execute a registered action by name.

    If the action is not found, returns a failure ActionResult.
    This is the single execution primitive — no recursion, no loops.
    """
    reg = registry or _default_registry
    fn = reg.get(action_name)
    if fn is None:
        return ActionResult(
            action_name=action_name,
            success=False,
            error=f"Action {action_name!r} not found in registry",
        )
    try:
        result = fn(context, **kwargs)
        if isinstance(result, ActionResult):
            return result
        # Assume the function returned a raw value — wrap it
        return ActionResult(
            action_name=action_name,
            success=True,
            output=result,
        )
    except Exception as exc:  # pragma: no cover
        return ActionResult(
            action_name=action_name,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# Convenience: bind actions to the global registry
def register_action(name: str) -> Callable[[SkillAction], SkillAction]:
    """Decorator: register a function as an action in the global registry."""

    def decorator(fn: SkillAction) -> SkillAction:
        _default_registry.register(name, fn)
        return fn

    return decorator
