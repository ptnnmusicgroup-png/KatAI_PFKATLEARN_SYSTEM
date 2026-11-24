"""Default-deny skill permission policy.

This policy authorizes only narrowly scoped skill operations. It does not grant
database, filesystem, network, shell, or administrative access.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str

SKILL_ACTIONS = {
    "KatWork": frozenset({"create_quiz"}),
    "KatGames": frozenset({"read_progress_summary"}),
    "KatVocab": frozenset({"lookup_word"}),
    "KatGrammar": frozenset({"explain_grammar"}),
    "KatTranslate": frozenset({"learning_translation"}),
}

def authorize_skill_action(skill: str, action: str) -> PermissionDecision:
    allowed_actions = SKILL_ACTIONS.get(skill)
    if allowed_actions is None:
        return PermissionDecision(False, "unknown_skill")
    if action not in allowed_actions:
        return PermissionDecision(False, "action_not_permitted")
    return PermissionDecision(True, "least_privilege_allowlist")
