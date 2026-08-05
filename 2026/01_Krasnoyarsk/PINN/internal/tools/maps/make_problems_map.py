"""Generate problems tag cloud PNG (WordCloud, 18:9) from translation dictionary."""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from numpy import linspace
from wordcloud import WordCloud

from translation_dict import LABEL_ENTRIES, fold, translation_stats

MAPS_DIR = Path(__file__).resolve().parent
ROOT = MAPS_DIR.parents[3]
PINN_DIR = ROOT
ASSETS = ROOT / "assets"
OUTPUT = ASSETS / "maps" / "problems.png"

CANVAS_W = 3840
CANVAS_H = 1920
DPI = 100

# Plasma: фиолетовый → розовый → оранжевый (без жёлтого хвоста), затемнённая для белого фона.
PLASMA_MAX = 0.74
PLASMA_DARKEN = 0.52
PLASMA_CMAP = ListedColormap(cm.plasma(linspace(0.0, PLASMA_MAX, 256)))
FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
]


def _resolve_font_path() -> str:
    for path in FONT_CANDIDATES:
        if path.exists():
            return str(path)
    return fm.findfont(fm.FontProperties(family="DejaVu Sans"))


FONT_PATH = _resolve_font_path()

BOX_DRAWING_RE = re.compile(r"[┌┐└┘├┤│─┼═╭╮╯╰]")
FORMULA_RE = re.compile(
    r"(=.*[Δλψωφδ∈]|^[Δλψωφδ]|ẋ|xₙ|dⁿy|Hψ|δJ|\bgλ\b|Kφ|\bAφ\b|"
    r"dy/dt|dx/dt|dⁿy/dt|/dt\s*=|=\s*[-0-9]|"
    r"\bF\s*\(|\bf\s*\(|\bg\s*\(|\[u\])",
    re.IGNORECASE,
)

NUMBER_PREFIX_RE = re.compile(r"^(?:\d+\.)+\s*")
SINGLE_NUMBER_RE = re.compile(r"^\d+\.\s+")

MAX_TAGS_PER_WEIGHT = 3
MIN_DISPLAY_WEIGHT = 15
MAX_DISPLAY_WEIGHT = 100

# Абстрактные/meta-подписи — не методы и не уравнения.
ABSTRACT_EXACT: frozenset[str] = frozenset({
    # общая деятельность
    "моделирование", "simulation", "prediction", "simulations",
    "моделирование химпроцессов", "chemical process simulation",
    "моделирование монте-карло", "monte carlo simulation",
    # meta-оси и цепочка
    "forward", "inverse", "forward problem", "inverse problem",
    "прямая задача", "обратная задача",
    "deterministic", "stochastic", "детерминированные", "стохастические",
    "управление", "control", "оптимизация", "optimization",
    "мультифизика", "multiphysics", "state problem",
    "идентификация параметров", "parameter identification",
    "стационарная / нестационарная", "прямая / обратная",
    "начальная / краевая / начально-краевая",
    "собственные значения",
    # гибриды и представления
    "цифровой двойник", "digital twin",
    "физика и ml", "physics + ml",
    "физика и эмпирические модели", "physics + empirical model",
    "континуальные и дискретные", "continuum + discrete",
    "гибридные физические модели", "digital twin / hybrid physics model",
    "processes", "applications", "процессы", "применения",
    "uncertain parameters", "robust", "probabilistic",
    "lumped / сосредоточенная", "continuum / сплошная среда",
    "distribution / распределение частиц", "particle / частицы",
    "discrete lattice", "network", "probabilistic state", "quantum state",
    "пространство", "случайность", "взаимодействие", "время", "масштаб", "постановка",
    "lumped / field deterministic local",
    "discrete / stochastic non-local",
    "continuous", "discrete", "delayed", "fractional",
    "quantum", "atomistic", "mesoscopic", "continuum", "macroscopic",
    "memory", "local", "non-local",
    # свойства без типа модели
    "linear", "nonlinear", "линейные", "нелинейные",
    "autonomous", "неавтономные",
    "стационарная", "нестационарная", "квазистационарная", "периодическая",
    "steady-state", "transient",
    "1. по времени", "2. по известным условиям", "3. по направлению вывода",
    "4. по искомой величине", "5. по математической цели", "6. по определённости",
    "по времени", "по известным условиям", "по направлению вывода",
    "по искомой величине", "по математической цели", "по определённости",
    "observations", "parameters", "state",
    "оценивание состояния", "идентификация по наблюдениям",
    "параметры к состояние", "наблюдения к параметры/источник/состояние",
})

# Заголовки разделов и meta-оси — не конкретные модели.
SECTION_HEADERS: frozenset[str] = frozenset({
    fold(x) for x in (
        "НЕПРЕРЫВНЫЕ МОДЕЛИ", "ДИСКРЕТНЫЕ МОДЕЛИ", "СТОХАСТИЧЕСКИЕ МОДЕЛИ",
        "КИНЕТИЧЕСКИЕ МОДЕЛИ", "ВАРИАЦИОННЫЕ МОДЕЛИ", "СПЕКТРАЛЬНЫЕ ЗАДАЧИ",
        "ОПТИМИЗАЦИОННЫЕ МОДЕЛИ", "ОБРАТНЫЕ И ИДЕНТИФИКАЦИОННЫЕ ЗАДАЧИ",
        "СЕТЕВЫЕ / ГРАФОВЫЕ МОДЕЛИ", "АГЕНТНЫЕ / ЧАСТИЧНЫЕ МОДЕЛИ",
        "МУЛЬТИФИЗИЧЕСКИЕ МОДЕЛИ", "МУЛЬТИМАСШТАБНЫЕ МОДЕЛИ", "ГИБРИДНЫЕ МОДЕЛИ",
        "Дифференциальные уравнения", "Интегральные уравнения",
        "По математическому типу", "Процессы", "Процессы:", "Применения",
        "По времени", "По известным условиям", "По направлению вывода",
        "По искомой величине", "По математической цели", "По определённости",
    )
})

# Приоритет при квантовании: известные термины — крупнее.
FAME_PRIORITY: dict[str, int] = {
    "оду": 1000,
    "учп": 1000,
    "навье-стокс": 990,
    "уравнение теплопроводности": 980,
    "уравнение диффузии": 975,
    "системы оду": 970,
    "системы учп": 970,
    "учп первого порядка": 968,
    "дробные оду": 965,
    "дробные учп": 965,
    "уравнения максвелла": 960,
    "мкэ": 950,
    "метод конечных элементов": 945,
    "вгд": 940,
    "монте-карло": 930,
    "молекулярная динамика": 925,
    "обратные задачи": 905,
    "реакция-диффузия": 900,
    "волновое уравнение": 895,
    "уравнение пуассона": 890,
    "уравнение лапласа": 885,
    "уравнения эйлера": 880,
    "дау": 875,
    "сду": 870,
    "сучп": 865,
    "марковские процессы": 860,
    "уравнение больцмана": 855,
    "решеточные уравнения больцмана": 850,
    "тфп": 840,
    "мгд": 835,
    "задача коши": 825,
    "краевая задача": 820,
    "метод конечных разностей": 815,
    "модели турбулентности": 810,
    "фазовое поле": 805,
    "цифровой двойник": 800,
}

SHORT_ABBREVS = frozenset({
    "ode", "pde", "fem", "cfd", "dae", "sde", "spde", "bvp", "ivp", "ibvp",
    "mhd", "fsi", "dft", "pinn", "abm", "dem", "sph", "kmc", "cme", "ae",
})

# Относительная «научная» частота (как в cloud.py: 15–100).
SCIENCE_FREQUENCY: dict[str, int] = {}
for weight, terms in {
    100: [
        "ODE", "PDE", "FEM", "CFD", "Navier–Stokes", "Navier-Stokes",
        "Heat equation",
        "Diffusion equation", "Maxwell", "Molecular Dynamics",
        "Monte Carlo", "DFT", "Inverse Problems", "Finite Element", "Finite Difference",
    ],
    92: [
        "DAE", "SDE", "SPDE", "BVP", "IVP", "IBVP", "Fokker–Planck",
        "Markov", "Boltzmann", "Reaction–Diffusion", "Reaction-Diffusion",
        "Multiphysics", "Fluid–Structure Interaction", "FSI", "MHD",
        "Poisson", "Laplace", "Wave equation", "Euler equations",
        "Optimal control", "Parameter identification", "Data Assimilation",
        "Heat conduction", "Mass diffusion", "Chemical kinetics", "Eigenvalue",
    ],
    84: [
        "DDE", "IDE", "Lattice Boltzmann", "Phase-field", "kMC",
        "Gillespie SSA", "Vlasov", "Brownian motion", "Langevin",
        "Galerkin", "FEM basis", "Topology Optimization", "Shape Optimization",
        "Poromechanics", "Thermo-mechanics", "Electrochemical transport",
        "Graph Laplacian", "Agent-Based Models", "ABM", "DEM", "SPH",
        "QM/MM", "Digital Twin", "Physics-Informed", "PINN",
    ],
    74: [
        "Fractional", "Functional Differential", "Integro-differential",
        "Boundary integral", "Variational", "Euler–Lagrange",
        "Master Equation", "CME", "Ising", "Cellular Automata",
        "Rarefied gases", "Plasma", "Semiconductor transport",
        "Stability analysis", "Bifurcation", "Modal Analysis",
        "Constitutive-law identification", "State Estimation",
    ],
    62: [
        "Van der Waals", "Peng–Robinson", "Ideal Gas", "Hamilton–Jacobi",
        "Tricomi", "Transonic", "Phonon", "Radiative Transfer",
        "Coarse-grained", "Homogenization", "Reactive flow",
        "Adsorption equilibrium", "Complementarity",
    ],
    48: [
        "Hereditary", "Memory kernels", "Anomalous diffusion",
        "Nonsmooth mechanics", "Grain growth", "Solidification",
        "Recrystallization", "Polymer dynamics", "Biological tissues",
    ],
    32: [],
    55: [
        "Forward Problem",
        "Inverse Problem",
    ],
}.items():
    for term in terms:
        SCIENCE_FREQUENCY[term.casefold()] = weight

KEYWORD_BOOSTS: list[tuple[int, tuple[str, ...]]] = [
    (95, ("navier", "stokes", "maxwell", "heat equation", "diffusion", "fem", "cfd", "molecular dynamics")),
    (88, ("monte carlo", "eigenvalue", "inverse", "optimization", "simulation", "markov", "boltzmann")),
    (78, ("sde", "dae", "spde", "bvp", "ivp", "multiphysics", "phase-field", "lattice boltzmann")),
    (68, ("fractional", "vlasov", "galerkin", "homogenization", "poromechanic", "thermo-")),
    (58, ("equilibrium", "transport", "kinetic", "viscoelastic", "porous")),
    (38, ("kernel", "hereditary", "tricomi", "recrystallization", "grain growth")),
]

GROUP_BASE: dict[str, int] = {
    "blue": 58,
    "green": 50,
    "orange": 52,
    "cyan": 46,
    "purple": 44,
    "magenta": 44,
    "red": 54,
    "volcano": 48,
    "geekblue": 42,
    "gold": 40,
    "lime": 50,
    "gray": 45,
    "chain": 52,
}


@dataclass(frozen=True)
class WeightedTag:
    label: str
    weight: int
    group: str


def strip_number_prefix(label: str) -> str:
    text = label.strip()
    text = NUMBER_PREFIX_RE.sub("", text)
    text = SINGLE_NUMBER_RE.sub("", text)
    return text.strip()


def merge_weighted_tags(tags: list[WeightedTag]) -> list[WeightedTag]:
    merged: dict[str, WeightedTag] = {}
    for tag in tags:
        clean = tag.label.strip()
        if not clean:
            continue
        key = clean.casefold()
        prev = merged.get(key)
        if prev is None or tag.weight > prev.weight:
            merged[key] = WeightedTag(clean, tag.weight, tag.group)
    return list(merged.values())


def is_concrete_label(text: str) -> bool:
    """Оставляем уравнения, методы и классы задач; отсекаем meta-абстракции."""
    key = fold(text)
    if key in ABSTRACT_EXACT:
        return False
    if key.endswith(" models") and key not in {
        "markov models", "convolution models", "graph laplacian models",
        "lattice models", "turbulence models", "модели монте-карло", "модели турбулентности",
    }:
        return False
    return True


def is_dictionary_entry(source_key: str, label: str) -> bool:
    if fold(source_key) in SECTION_HEADERS:
        return False
    if fold(label) in SECTION_HEADERS:
        return False
    return is_concrete_label(label) and is_renderable_label(label)


def is_renderable_label(text: str) -> bool:
    if not is_concrete_label(text):
        return False
    if not text or len(text) > 72:
        return False
    if BOX_DRAWING_RE.search(text):
        return False
    if FORMULA_RE.search(text):
        return False
    if text.count("=") >= 1 and sum(ch.isalpha() for ch in text) < 4:
        return False
    if text.endswith(":"):
        return False
    if sum(ch.isalpha() for ch in text) < 3:
        return False
    return True


def fame_score(label: str) -> int:
    return FAME_PRIORITY.get(label.casefold(), 0)


def _term_matches(key: str, term: str) -> bool:
    if key == term:
        return True
    if len(term) <= 4 and term not in SHORT_ABBREVS:
        return False
    if len(term) >= 5:
        return term in key or key in term
    # Короткие аббревиатуры — только как отдельное слово.
    return bool(re.search(rf"(?<![a-zа-яё0-9]){re.escape(term)}(?![a-zа-яё0-9])", key))


def science_weight(label: str, group: str) -> int:
    clean = strip_number_prefix(label)
    if not clean or clean in {"│", "├", "└"}:
        return 0

    key = clean.casefold()
    if key in SCIENCE_FREQUENCY:
        return SCIENCE_FREQUENCY[key]

    for term, weight in SCIENCE_FREQUENCY.items():
        if _term_matches(key, term):
            return weight

    best = GROUP_BASE.get(group, 35)
    for boost, words in KEYWORD_BOOSTS:
        if any(w in key for w in words):
            best = max(best, boost)

    if re.fullmatch(r"[A-ZА-Я]{2,6}", clean):
        best = max(best, 65)

    if re.search(r"[=Δ∈≥≤/]", clean):
        best = max(best, 40)

    return best


def quantize_frequencies(
    tags: list[WeightedTag],
    *,
    max_per_weight: int = MAX_TAGS_PER_WEIGHT,
    max_weight: int = MAX_DISPLAY_WEIGHT,
    min_weight: int = MIN_DISPLAY_WEIGHT,
) -> tuple[dict[str, int], dict[str, str]]:
    """Не более max_per_weight тегов на одно значение частоты."""
    ranked = sorted(
        tags,
        key=lambda t: (-t.weight, -fame_score(t.label), len(t.label), t.label.casefold()),
    )
    num_tiers = max(1, (len(ranked) + max_per_weight - 1) // max_per_weight)

    frequencies: dict[str, int] = {}
    groups: dict[str, str] = {}
    for rank, tag in enumerate(ranked):
        tier = rank // max_per_weight
        if num_tiers == 1:
            display_weight = max_weight
        else:
            display_weight = max_weight - tier * (max_weight - min_weight) / (num_tiers - 1)
        display_weight = round(display_weight, 1)

        if tag.label not in frequencies or frequencies[tag.label] < display_weight:
            frequencies[tag.label] = display_weight
            groups[tag.label] = tag.group

    return frequencies, groups


def build_plasma_colors(frequencies: dict[str, float]) -> dict[str, str]:
    """Plasma по словам (hash), затемнённая для контраста на белом фоне."""
    colors: dict[str, str] = {}
    for word in frequencies:
        t = (zlib.adler32(word.casefold().encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF
        red, green, blue, _alpha = PLASMA_CMAP(t)
        colors[word] = mcolors.to_hex(
            (red * PLASMA_DARKEN, green * PLASMA_DARKEN, blue * PLASMA_DARKEN),
            keep_alpha=False,
        )
    return colors


def collect_all_tags() -> tuple[dict[str, float], dict[str, str]]:
    """Собирает теги напрямую из LABEL_ENTRIES (русские значения словаря)."""
    raw: list[WeightedTag] = []
    for source_key, label in LABEL_ENTRIES.items():
        if not is_dictionary_entry(source_key, label):
            continue
        weight = science_weight(source_key, "gray")
        if weight <= 0:
            continue
        raw.append(WeightedTag(label, weight, "gray"))
    return quantize_frequencies(merge_weighted_tags(raw))


def render_problems_cloud(output: Path = OUTPUT) -> Path:
    frequencies, groups = collect_all_tags()
    word_colors = build_plasma_colors(frequencies)

    def color_func(word: str, **_kwargs) -> str:
        return word_colors.get(word, "#595959")

    cloud = WordCloud(
        width=CANVAS_W,
        height=CANVAS_H,
        background_color="white",
        font_path=FONT_PATH,
        prefer_horizontal=0.75,
        color_func=color_func,
        random_state=42,
        min_font_size=10,
        max_font_size=180,
        collocations=False,
    ).generate_from_frequencies(frequencies)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(CANVAS_W / DPI, CANVAS_H / DPI), dpi=DPI)
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output, dpi=DPI, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close()
    return output


def main() -> None:
    output = render_problems_cloud()
    frequencies, _ = collect_all_tags()
    top = sorted(frequencies.items(), key=lambda x: -x[1])[:12]
    tier_sizes = Counter(frequencies.values())
    print(f"Dictionary: {translation_stats()['labels']} labels")
    print(f"Saved: {output}")
    print(f"Canvas: {CANVAS_W}×{CANVAS_H} (18:9)")
    print(f"Tags: {len(frequencies)}")
    print(f"Unique weights: {len(tier_sizes)} (max per weight: {max(tier_sizes.values())})")
    print("Top by display weight:", ", ".join(f"{k}({v})" for k, v in top))


if __name__ == "__main__":
    main()
