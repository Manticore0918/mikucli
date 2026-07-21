from __future__ import annotations

import re


SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
SKILL_PREFIX_PATTERN = re.compile(r"^\$([a-z][a-z0-9]*(?:-[a-z0-9]+)*)(?:\s+|$)")
MAX_SKILL_NAME_LENGTH = 64


def valid_skill_name(name: str) -> bool:
    return len(name) <= MAX_SKILL_NAME_LENGTH and SKILL_NAME_PATTERN.fullmatch(name) is not None


def parse_skill_prefix(prompt: str) -> tuple[str, str] | None:
    match = SKILL_PREFIX_PATTERN.match(prompt)
    if match is None:
        return None
    name = match.group(1)
    if not valid_skill_name(name):
        return None
    return name, prompt[match.end() :].strip()

