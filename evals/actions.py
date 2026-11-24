"""Shared action canonicalization for free-text ``physical_action`` values.

Live runs should constrain ``physical_action`` with the scenario's action enum (see
``LlamaCppBackend(action_space=...)``), which makes this unnecessary. This module exists for
free-text output and legacy reports. It never guesses: text matching no option is
``other_action``, text matching several options is ``ambiguous``, empty text is ``unparsed``.
"""

from collections.abc import Mapping

AMBIGUOUS = "ambiguous"
OTHER = "other_action"
UNPARSED = "unparsed"


def canonicalize(raw_action: str, options: Mapping[str, tuple[str, ...]]) -> str:
    norm = " ".join(raw_action.strip().lower().replace("_", " ").split())
    if not norm:
        return UNPARSED
    for label in options:  # exact label (enum-constrained output) wins outright
        if norm == label.replace("_", " "):
            return label
    hits = [
        label
        for label, keywords in options.items()
        if any(keyword.replace("_", " ") in norm for keyword in keywords)
    ]
    if len(hits) == 1:
        return hits[0]
    return AMBIGUOUS if hits else OTHER
