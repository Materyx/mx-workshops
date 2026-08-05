"""Parse tree-structured taxonomy from context.txt."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

BRANCH_RE = re.compile(r"^(?P<prefix>[\s│]*)(?P<marker>├──|└──)\s*(?P<label>.+)$")
SECTION_HEADERS = {
    "ТИП ПОСТАНОВКИ ЗАДАЧИ",
    "ПРЕДСТАВЛЕНИЕ ФИЗИЧЕСКОЙ СИСТЕМЫ",
}


@dataclass
class TreeNode:
    label: str
    children: list[TreeNode] = field(default_factory=list)

    def flatten(self, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        lines = [f"{prefix}{self.label}"]
        for child in self.children:
            lines.extend(child.flatten(indent + 1))
        return lines

    def line_count(self) -> int:
        return len(self.flatten())


@dataclass
class ContextDocument:
    title: str
    subtitle: str
    chain: str
    models: TreeNode
    problem_setup: TreeNode
    representation: TreeNode
    meta_axes: list[str]


def _depth_from_prefix(prefix: str) -> int:
    if "│" in prefix:
        pipes = prefix.count("│")
        tail = prefix.split("│")[-1]
        return pipes + len(tail) // 4
    return len(prefix) // 4


def _parse_tree_lines(lines: list[str]) -> TreeNode:
    root = TreeNode(label="__root__")
    stack: list[tuple[int, TreeNode]] = [(-1, root)]

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.strip() == "│":
            continue

        match = BRANCH_RE.match(line)
        if not match:
            continue

        depth = _depth_from_prefix(match.group("prefix"))
        label = match.group("label").strip()
        node = TreeNode(label=label)

        while stack and stack[-1][0] >= depth:
            stack.pop()

        parent = stack[-1][1]
        parent.children.append(node)
        stack.append((depth, node))

    return root


def _extract_top_sections(models_root: TreeNode) -> list[TreeNode]:
    return models_root.children


def parse_context_file(path: Path | str) -> ContextDocument:
    text = Path(path).read_text(encoding="utf-8")
    raw_lines = text.splitlines()

    title = raw_lines[0].strip()
    subtitle = raw_lines[1].strip()

    models_lines: list[str] = []
    setup_lines: list[str] = []
    repr_lines: list[str] = []
    meta_lines: list[str] = []
    chain = ""

    section = "models"
    for line in raw_lines[2:]:
        stripped = line.strip()
        if stripped.startswith("стационарная /"):
            chain = stripped
            section = "between"
            continue
        if stripped == "ТИП ПОСТАНОВКИ ЗАДАЧИ":
            section = "setup"
            continue
        if stripped == "ПРЕДСТАВЛЕНИЕ ФИЗИЧЕСКОЙ СИСТЕМЫ":
            section = "representation"
            continue
        if stripped.startswith("ФИЗИКО-ХИМИЧЕСКАЯ МОДЕЛЬ"):
            section = "meta"
            meta_lines.append(line.rstrip())
            continue

        if section == "models":
            models_lines.append(line)
        elif section == "setup":
            setup_lines.append(line)
        elif section == "representation":
            repr_lines.append(line)
        elif section == "meta":
            meta_lines.append(line.rstrip())

    models_root = _parse_tree_lines(models_lines)
    setup_root = _parse_tree_lines(setup_lines)
    repr_root = _parse_tree_lines(repr_lines)

    meta_clean = []
    for line in meta_lines:
        cleaned = line.strip()
        if cleaned and cleaned not in {"│", "├", "└"}:
            meta_clean.append(cleaned)

    return ContextDocument(
        title=title,
        subtitle=subtitle,
        chain=chain,
        models=models_root,
        problem_setup=setup_root,
        representation=repr_root,
        meta_axes=meta_clean,
    )


def load_default_context() -> ContextDocument:
    path = Path(__file__).with_name("context.txt")
    return parse_context_file(path)
