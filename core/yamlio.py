#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/yamlio.py

Tiny YAML subset loader/dumper (no PyYAML dependency).

Why we keep a tiny YAML subset
------------------------------
PyYAML is convenient, but it adds a dependency that can be annoying in:
- minimal server deployments
- student environments
- reproducibility constraints

So this file implements *just enough* YAML for the config files in this repo.

Supported YAML subset
---------------------
1) Dict (mapping) scalars:
    key: value
    key: "quoted string"
    key: true/false
    key: 123
    key: 1.23
    key: null

2) Nested dicts (2-space indentation):
    parent:
      child: value
      nested:
        k: v

3) Lists:
    items:
      - 1
      - "two"
      - true

4) List of dicts (common config style):
    ops:
      - code: OP-1
        tau_high: 0.72
        tau_low_ratio: 0.8

Notes / limitations (intentional)
---------------------------------
- Indentation must be spaces (tabs are rejected).
- We don’t support complex YAML features (anchors, multiline blocks, etc.).
- Keys must be simple (no ":" inside keys).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple, Union


# ============================================================
# 1) Scalar parsing helpers
# ============================================================
# Regexes help us decide whether a value looks like an int/float.
_RE_INT = re.compile(r"[-+]?\d+\Z")
_RE_FLOAT = re.compile(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?\Z|[-+]?\d+[eE][-+]?\d+\Z")


def _strip_inline_comment(line: str) -> str:
    """
    Remove inline comments starting with '#', but only if the '#' is not inside quotes.

    Examples:
      key: 1  # comment      -> "key: 1"
      key: "a#b"             -> stays the same (comment NOT removed)
      key: 'a#b'             -> stays the same

    This is a simple state machine:
      - track whether we are inside single quotes or double quotes
      - if not inside quotes, '#' begins a comment
    """
    in_s = False  # inside '...'
    in_d = False  # inside "..."
    out_chars: List[str] = []

    for ch in line:
        if ch == "'" and not in_d:
            in_s = not in_s
            out_chars.append(ch)
            continue
        if ch == '"' and not in_s:
            in_d = not in_d
            out_chars.append(ch)
            continue

        if ch == "#" and not in_s and not in_d:
            break  # start of comment -> stop copying
        out_chars.append(ch)

    return "".join(out_chars).rstrip()


def _parse_scalar(s: str) -> Any:
    """
    Parse a YAML scalar value into Python types.

    Supported:
      - true/false -> bool
      - null/none  -> None
      - int/float  -> numeric
      - quoted str -> str (quotes removed)
      - otherwise  -> raw string

    This function is intentionally small and predictable.
    """
    s = s.strip()
    if not s:
        return ""

    lo = s.lower()

    # Booleans
    if lo in {"true", "false"}:
        return lo == "true"

    # Null
    if lo in {"null", "none"}:
        return None

    # Numbers
    if _RE_INT.match(s):
        try:
            return int(s)
        except Exception:
            return s

    if _RE_FLOAT.match(s):
        try:
            return float(s)
        except Exception:
            return s

    # Quoted strings:
    # We accept both single and double quotes.
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]

    return s


# ============================================================
# 2) Tokenize lines with indentation
# ============================================================
Token = Tuple[int, str, int]  # (indent_spaces, content, line_number)


def _tokenize_lines(lines: Sequence[str]) -> List[Token]:
    """
    Convert raw file lines into meaningful tokens.

    Steps:
      1) Reject tabs (we require spaces)
      2) Strip comments (respect quotes)
      3) Skip empty lines
      4) Measure indentation (# leading spaces)
      5) Keep (indent, content, line_no)

    Output tokens are in original order.
    """
    tokens: List[Token] = []

    for i, raw in enumerate(lines, start=1):
        if "\t" in raw:
            raise ValueError(f"YAML parse error at line {i}: tabs are not allowed. Use spaces.")

        # Remove newline and strip inline comments safely
        line = raw.rstrip("\n")
        line = _strip_inline_comment(line)

        # Skip empty/comment-only lines
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        tokens.append((indent, content, i))

    return tokens


# ============================================================
# 3) Recursive descent parser (dict + list)
# ============================================================
YamlObj = Union[Dict[str, Any], List[Any]]


def _expect_indent(indent: int, expected: int, line_no: int) -> None:
    """
    Enforce indentation levels.

    We accept any indent values as long as nesting is consistent,
    but we encourage 2-space increments by how we descend.
    """
    if indent < expected:
        raise ValueError(f"YAML parse error at line {line_no}: indentation too small (got {indent}, expected >= {expected}).")


def _is_list_item(content: str) -> bool:
    """Return True if this token line begins a list item '- ...'."""
    return content.startswith("-") and (content == "-" or content.startswith("- "))


def _parse_block(tokens: List[Token], i: int, indent_level: int) -> Tuple[YamlObj, int]:
    """
    Parse a block at a specific indentation level.

    The block can be:
      - a dict block (lines like 'key: value')
      - a list block (lines like '- item')

    Returns:
      (obj, next_index)
    """
    if i >= len(tokens):
        return {}, i

    indent, content, line_no = tokens[i]
    if indent < indent_level:
        # caller will handle this as "end of block"
        return {}, i

    # Decide whether this block is a list or dict based on the first line.
    if indent == indent_level and _is_list_item(content):
        return _parse_list(tokens, i, indent_level)
    return _parse_dict(tokens, i, indent_level)


def _parse_dict(tokens: List[Token], i: int, indent_level: int) -> Tuple[Dict[str, Any], int]:
    """
    Parse a dict block starting at tokens[i] with exact indent_level.

    Dict line formats:
      key: value
      key:
        nested...

    Returns:
      (dict_obj, next_index)
    """
    out: Dict[str, Any] = {}

    while i < len(tokens):
        indent, content, line_no = tokens[i]

        # End of this dict block
        if indent < indent_level:
            break

        # Lines at deeper indentation belong to a nested parse started earlier
        if indent > indent_level:
            raise ValueError(f"YAML parse error at line {line_no}: unexpected extra indentation.")

        # Dict key line must contain ':'
        if ":" not in content:
            raise ValueError(f"YAML parse error at line {line_no}: expected 'key: value' but got: {content!r}")

        key, val = content.split(":", 1)
        key = key.strip()
        val = val.strip()

        if not key:
            raise ValueError(f"YAML parse error at line {line_no}: empty key is not allowed.")

        # Case A: key: (empty) -> nested block (dict or list)
        if val == "":
            # Look ahead to decide nested container type
            j = i + 1
            if j >= len(tokens) or tokens[j][0] <= indent_level:
                # No nested lines -> treat as empty dict
                out[key] = {}
                i += 1
                continue

            # Parse nested block at indent_level+2
            obj, j2 = _parse_block(tokens, j, indent_level + 2)
            out[key] = obj
            i = j2
            continue

        # Case B: key: scalar
        out[key] = _parse_scalar(val)
        i += 1

    return out, i


def _parse_list(tokens: List[Token], i: int, indent_level: int) -> Tuple[List[Any], int]:
    """
    Parse a list block starting at tokens[i] with exact indent_level.

    List line formats:
      - value
      - key: value        (inline dict item)
      -                 (empty item -> nested block on following lines)

    Returns:
      (list_obj, next_index)
    """
    out: List[Any] = []

    while i < len(tokens):
        indent, content, line_no = tokens[i]

        # End of this list block
        if indent < indent_level:
            break

        if indent > indent_level:
            raise ValueError(f"YAML parse error at line {line_no}: unexpected extra indentation in list.")

        if not _is_list_item(content):
            # At the same indent level, list must continue with '-' items
            break

        item_text = content[1:].strip()  # remove leading '-'
        item_indent = indent_level + 2   # nested lines for this list item are here

        # Case 1: "- " with no text -> nested object is required on following lines
        if item_text == "":
            j = i + 1
            if j >= len(tokens) or tokens[j][0] <= indent_level:
                # Empty item with no nested block -> treat as empty dict
                out.append({})
                i += 1
                continue
            obj, j2 = _parse_block(tokens, j, item_indent)
            out.append(obj)
            i = j2
            continue

        # Case 2: "- key: value" (inline dict head)
        # This is common in config: a list of dict objects.
        if ":" in item_text:
            k0, v0 = item_text.split(":", 1)
            k0 = k0.strip()
            v0 = v0.strip()

            item_dict: Dict[str, Any] = {}
            if v0 == "":
                # "- key:" -> parse nested block for that key
                j = i + 1
                if j < len(tokens) and tokens[j][0] > indent_level:
                    nested, j2 = _parse_block(tokens, j, item_indent + 2)
                    item_dict[k0] = nested
                    i = j2
                else:
                    item_dict[k0] = {}
                    i += 1
            else:
                item_dict[k0] = _parse_scalar(v0)
                i += 1

            # Now parse additional dict fields belonging to this list item.
            # They appear as:
            #   - code: OP-1
            #     tau_high: 0.7
            #     tau_low_ratio: 0.8
            # Those extra keys are at indent=item_indent.
            if i < len(tokens) and tokens[i][0] == item_indent and not _is_list_item(tokens[i][1]):
                extra, i2 = _parse_dict(tokens, i, item_indent)
                item_dict.update(extra)
                i = i2

            out.append(item_dict)
            continue

        # Case 3: "- scalar"
        out.append(_parse_scalar(item_text))
        i += 1

    return out, i


# ============================================================
# 4) Public API: load
# ============================================================
def yaml_load_simple(path: str) -> Dict[str, Any]:
    """
    Load a small YAML subset into nested Python objects.

    Returns:
      dict at the top-level (like most YAML config files)

    Raises:
      ValueError with line-number info on parse problems.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tokens = _tokenize_lines(lines)
    if not tokens:
        return {}

    obj, next_i = _parse_block(tokens, 0, tokens[0][0])

    # Enforce that root is a dict for config files
    if not isinstance(obj, dict):
        indent, content, line_no = tokens[0]
        raise ValueError(
            f"YAML parse error at line {line_no}: top-level must be a dict (key: value), "
            f"but got a list item: {content!r}"
        )

    # If there are leftover tokens, it usually indicates indentation/format errors
    if next_i != len(tokens):
        indent, content, line_no = tokens[next_i]
        raise ValueError(f"YAML parse error at line {line_no}: could not parse remaining content: {content!r}")

    return obj


# ============================================================
# 5) Public API: dump
# ============================================================
def yaml_dump_simple(obj: Dict[str, Any], path: str) -> None:
    """
    Write a YAML file using the same small subset.

    Supports dict + list recursively.

    Strings are written using JSON quoting (json.dumps) because:
    - it safely escapes quotes and special characters
    - it keeps unicode readable with ensure_ascii=False
    """
    def dump_value(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            # For floats, Python str() is OK for configs.
            return str(v)
        # Quote strings safely
        return json.dumps(str(v), ensure_ascii=False)

    lines: List[str] = []

    def rec(node: Any, indent: int) -> None:
        """
        Recursive dumper.

        node can be:
          - dict: write 'key:' lines
          - list: write '- item' lines
          - scalar: write as value
        """
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    lines.append(" " * indent + f"{k}:")
                    rec(v, indent + 2)
                else:
                    lines.append(" " * indent + f"{k}: {dump_value(v)}")
            return

        if isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    lines.append(" " * indent + "-")
                    rec(item, indent + 2)
                else:
                    lines.append(" " * indent + f"- {dump_value(item)}")
            return

        # Scalar fallback (rare at root): still write something meaningful
        lines.append(" " * indent + dump_value(node))

    rec(obj, 0)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
