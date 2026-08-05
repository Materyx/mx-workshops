"""Сгенерировать синтетические csv для workshop (аналитика + шум).

Запуск:
    python3 internal/tools/make_workshop_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "workshop" / "data"
DATA.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


def save_csv(path: Path, t: np.ndarray, y: np.ndarray, y_name: str = "y") -> None:
    header = f"t,{y_name}"
    np.savetxt(path, np.column_stack([t, y]), delimiter=",", header=header, comments="")


def save_xyz_csv(path: Path, x: np.ndarray, y: np.ndarray, c: np.ndarray) -> None:
    header = "x,y,c"
    np.savetxt(path, np.column_stack([x, y, c]), delimiter=",", header=header, comments="")


def make_free_fall() -> None:
    g, y0, v0 = 9.8, 1.0, 4.0
    t = np.linspace(0.0, 0.8, 25)
    y = y0 + v0 * t - 0.5 * g * t**2
    y = y + RNG.normal(0.0, 0.03, size=t.shape)
    save_csv(DATA / "free_fall.csv", t.astype(np.float64), y.astype(np.float64), "y")


def make_harmonic() -> None:
    omega, A, B = 2.0, 1.0, 0.3
    t = np.linspace(0.0, 2.0 * np.pi, 40)
    x = A * np.cos(omega * t) + B * np.sin(omega * t)
    x = x + RNG.normal(0.0, 0.05, size=t.shape)
    save_csv(DATA / "harmonic_oscillator.csv", t.astype(np.float64), x.astype(np.float64), "x")


def make_first_order() -> None:
    k, c0 = 0.8, 1.0
    t = np.linspace(0.0, 4.0, 30)
    c = c0 * np.exp(-k * t)
    c = c + RNG.normal(0.0, 0.02, size=t.shape)
    c = np.clip(c, 0.0, None)
    save_csv(DATA / "first_order_kinetics.csv", t.astype(np.float64), c.astype(np.float64), "c")


def make_reversible() -> None:
    k1, k_1, a0 = 1.2, 0.4, 1.0
    ksum = k1 + k_1
    a_eq = k_1 * a0 / ksum
    t = np.linspace(0.0, 5.0, 35)
    a = a_eq + (a0 - a_eq) * np.exp(-ksum * t)
    a = a + RNG.normal(0.0, 0.02, size=t.shape)
    a = np.clip(a, 0.0, a0)
    save_csv(DATA / "reversible_reaction.csv", t.astype(np.float64), a.astype(np.float64), "a")


def _sample_interior(n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Случайные точки строго внутри (0, 1)²."""
    x = RNG.uniform(0.05, 0.95, size=n)
    y = RNG.uniform(0.05, 0.95, size=n)
    return x, y


def make_diffusion_reaction_2d() -> None:
    """c = sin(πx)sin(πy), −∇²c + k c = f."""
    x, y = _sample_interior(80)
    c = np.sin(np.pi * x) * np.sin(np.pi * y)
    c = c + RNG.normal(0.0, 0.02, size=c.shape)
    save_xyz_csv(
        DATA / "diffusion_reaction_2d.csv",
        x.astype(np.float64),
        y.astype(np.float64),
        c.astype(np.float64),
    )


def make_diffusion_source_2d() -> None:
    """c = sin(πx)sin(2πy), −∇²c = S."""
    x, y = _sample_interior(80)
    c = np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)
    c = c + RNG.normal(0.0, 0.02, size=c.shape)
    save_xyz_csv(
        DATA / "diffusion_source_2d.csv",
        x.astype(np.float64),
        y.astype(np.float64),
        c.astype(np.float64),
    )


if __name__ == "__main__":
    make_free_fall()
    make_harmonic()
    make_first_order()
    make_reversible()
    make_diffusion_reaction_2d()
    make_diffusion_source_2d()
    print(f"Данные записаны в {DATA}")
    for path in sorted(DATA.glob("*.csv")):
        print(f"  {path.name}")
