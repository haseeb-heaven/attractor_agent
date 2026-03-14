"""Condition expression language for edge guards.

Implements: key=value, key!=value, && conjunction.
Variables: outcome, preferred_label, context.*.
"""

from __future__ import annotations

from attractor.pipeline.context import Context, Outcome


def resolve_key(key: str, outcome: Outcome, context: Context) -> str:
    """Resolve a variable key to its string value."""
    key = key.strip()

    if key == "outcome":
        return outcome.status.value

    if key == "preferred_label":
        return outcome.preferred_label

    if key.startswith("context."):
        # Try with prefix
        value = context.get(key)
        if value is not None:
            return str(value)
        # Try without prefix
        bare_key = key[len("context."):]
        value = context.get(bare_key)
        if value is not None:
            return str(value)
        return ""

    # Direct context lookup for unqualified keys
    value = context.get(key)
    if value is not None:
        return str(value)
    return ""


def evaluate_clause(clause: str, outcome: Outcome, context: Context) -> bool:
    """Evaluate a single clause (key=value or key!=value)."""
    clause = clause.strip()
    if not clause:
        return True

    if "!=" in clause:
        parts = clause.split("!=", 1)
        key = parts[0].strip()
        value = parts[1].strip()
        return resolve_key(key, outcome, context) != value

    if "=" in clause:
        parts = clause.split("=", 1)
        key = parts[0].strip()
        value = parts[1].strip()
        return resolve_key(key, outcome, context) == value

    # Bare key: check if truthy
    resolved = resolve_key(clause, outcome, context)
    return bool(resolved) and resolved.lower() not in ("false", "0", "")


def evaluate_condition(condition: str, outcome: Outcome, context: Context) -> bool:
    """Evaluate a full condition expression.

    Supports && conjunction of clauses.
    Empty condition always returns True (unconditional edge).
    """
    if not condition or not condition.strip():
        return True

    clauses = condition.split("&&")
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if not evaluate_clause(clause, outcome, context):
            return False
    return True
