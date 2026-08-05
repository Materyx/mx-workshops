"""Build tag-cloud HTML poster (18:9) from context.txt."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from parse_context import ContextDocument, TreeNode, load_default_context

MAPS_DIR = Path(__file__).resolve().parent

SECTION_GROUPS = [
    "blue", "green", "orange", "cyan", "purple", "magenta",
    "red", "volcano", "geekblue", "gold", "lime", "geekblue", "gray",
]

SECTION_LEGEND = [
    ("01", "Непрерывные", "blue"),
    ("02", "Дискретные", "green"),
    ("03", "Стохастические", "orange"),
    ("04", "Кинетические", "cyan"),
    ("05", "Вариационные", "purple"),
    ("06", "Спектральные", "magenta"),
    ("07", "Оптимизация", "red"),
    ("08", "Обратные", "volcano"),
    ("09", "Сетевые", "geekblue"),
    ("10", "Агентные", "gold"),
    ("11", "Мультифизика", "lime"),
    ("12", "Мультимасштаб", "geekblue"),
    ("13", "Гибридные", "gray"),
]

FORMULA_RE = re.compile(r"[=Δ∈≥≤λφψω∂×/]|d[nⁿ]?y/dt|dx/dt|min J|δJ")


@dataclass(frozen=True)
class TagItem:
    label: str
    group: str
    size: str  # lg | md | sm
    mono: bool = False


def is_formula(label: str) -> bool:
    return bool(FORMULA_RE.search(label))


def collect_tree_tags(node: TreeNode, group: str, depth: int = 0) -> list[TagItem]:
    tags: list[TagItem] = []
    if node.label and node.label != "__root__":
        size = "lg" if depth == 0 else ("md" if depth == 1 else "sm")
        tags.append(TagItem(node.label, group, size, is_formula(node.label)))
    for child in node.children:
        tags.extend(collect_tree_tags(child, group, depth + 1))
    return tags


def collect_document_tags(doc: ContextDocument) -> dict[str, list[TagItem]]:
    pages: dict[str, list[TagItem]] = {"page-1": [], "page-2": [], "page-3": []}

    sec0 = doc.models.children[0]
    pages["page-1"] = collect_tree_tags(sec0, SECTION_GROUPS[0])

    for idx, section in enumerate(doc.models.children[1:], start=1):
        pages["page-2"].extend(collect_tree_tags(section, SECTION_GROUPS[idx]))

    pages["page-3"].extend(collect_tree_tags(doc.problem_setup, "gray"))
    pages["page-3"].extend(collect_tree_tags(doc.representation, "gray"))

    for line in doc.meta_axes:
        cleaned = line.strip("│├└─ ")
        if cleaned and not cleaned.startswith("ФИЗИКО-ХИМИЧЕСКАЯ"):
            pages["page-3"].append(TagItem(cleaned, "gray", "sm"))

    for step in [s.strip().rstrip(".") for s in doc.chain.split("→") if s.strip()]:
        pages["page-3"].append(TagItem(step, "chain", "md"))

    return pages


def render_tag(tag: TagItem) -> str:
    mono = " tag-item--mono" if tag.mono else ""
    return (
        f'<span class="tag-item tag-item--{tag.size}{mono}" '
        f'data-group="{html.escape(tag.group, quote=True)}">'
        f"{html.escape(tag.label)}</span>"
    )


def render_tag_cloud(tags: list[TagItem]) -> str:
    return f'<div class="tag-cloud">{"".join(render_tag(t) for t in tags)}</div>'


def render_legend(groups: list[tuple[str, str, str]]) -> str:
    items = "".join(
        f'<span class="legend__item">'
        f'<span class="legend__swatch" data-group="{g}"></span>{html.escape(name)}</span>'
        for _, name, g in groups
    )
    return f'<div class="legend">{items}</div>'


def _page_shell(
    doc: ContextDocument,
    page_id: str,
    subtitle: str,
    tags: list[TagItem],
    legend: list[tuple[str, str, str]] | None,
    page_num: int,
    total: int,
) -> str:
    legend_html = render_legend(legend) if legend else ""
    return f"""
<main class="poster" id="{page_id}">
  <header class="hero">
    <div>
      <h1 class="hero__title">{html.escape(doc.title)}</h1>
      <p class="hero__subtitle">{html.escape(subtitle)}</p>
    </div>
    <span class="page-badge">{page_num} / {total}</span>
  </header>
  {legend_html}
  {render_tag_cloud(tags)}
</main>
"""


def build_html(doc: ContextDocument, css_href: str = "problems.css") -> str:
    tag_pages = collect_document_tags(doc)
    total = 3

    page1 = _page_shell(
        doc, "page-1", "Непрерывные модели · облако тегов",
        tag_pages["page-1"], [SECTION_LEGEND[0]], 1, total,
    )
    page2 = _page_shell(
        doc, "page-2", "Классы моделей 02–13 · облако тегов",
        tag_pages["page-2"], SECTION_LEGEND[1:], 2, total,
    )
    page3 = _page_shell(
        doc, "page-3", "Постановка · представление · мета-оси · облако тегов",
        tag_pages["page-3"],
        [("—", "Постановка / представление", "gray"), ("→", "Цепочка классификации", "chain")],
        3, total,
    )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>{html.escape(doc.title)}</title>
  <link rel="stylesheet" href="{html.escape(css_href)}">
</head>
<body>
  {page1}
  {page2}
  {page3}
</body>
</html>
"""


def write_html(doc: ContextDocument | None = None, output: Path | None = None) -> Path:
    doc = doc or load_default_context()
    output = output or MAPS_DIR / "problems.html"
    output.write_text(build_html(doc), encoding="utf-8")
    return output
