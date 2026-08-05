"""Общие пути и хелперы для workshop-ноутбуков."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "workshop" / "data"
ASSETS = ROOT / "assets"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


class MLP(nn.Module):
    """Маленькая полносвязная сеть u(t) -> R."""

    def __init__(self, n_hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class MLP2D(nn.Module):
    """Полносвязная сеть c(x, y) -> R."""

    def __init__(self, n_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


def laplacian(u: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """∇²u для u(x,y), где xy имеет shape (N, 2)."""
    grad_u = derivative(u, xy)
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = derivative(u_x, xy)[:, 0:1]
    u_yy = derivative(u_y, xy)[:, 1:2]
    return u_xx + u_yy


def load_xy_csv(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = table.dtype.names
    if names is None or len(names) < 2:
        raise ValueError(f"Ожидались колонки t,y в {path}")
    t = np.asarray(table[names[0]], dtype=np.float32)
    y = np.asarray(table[names[1]], dtype=np.float32)
    return t, y


def load_xyz_csv(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Загрузить колонки x, y, c из csv."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = table.dtype.names
    if names is None or len(names) < 3:
        raise ValueError(f"Ожидались колонки x,y,c в {path}")
    x = np.asarray(table[names[0]], dtype=np.float32)
    y = np.asarray(table[names[1]], dtype=np.float32)
    c = np.asarray(table[names[2]], dtype=np.float32)
    return x, y, c


def plot_solution(
    t_data: np.ndarray,
    y_data: np.ndarray,
    t_grid: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.scatter(t_data, y_data, color="red", s=22, label="Данные", zorder=3)
    plt.plot(t_grid, y_true, "k--", linewidth=1.6, label="Аналитика")
    plt.plot(t_grid, y_pred, "b", linewidth=1.4, label="PINN")
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_field_2d(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    c_true: np.ndarray,
    c_pred: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
) -> None:
    """Три heatmap: аналитика | PINN | |ошибка|; поверх — точки данных."""
    err = np.abs(c_pred - c_true)
    vmin = float(min(c_true.min(), c_pred.min()))
    vmax = float(max(c_true.max(), c_pred.max()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    panels = (
        (c_true, "Аналитика", vmin, vmax, cmap),
        (c_pred, "PINN", vmin, vmax, cmap),
        (err, "|Ошибка|", 0.0, float(err.max() + 1e-12), "magma"),
    )
    for ax, (field, label, lo, hi, cm) in zip(axes, panels):
        pcm = ax.pcolormesh(x_grid, y_grid, field, shading="auto", cmap=cm, vmin=lo, vmax=hi)
        ax.scatter(x_data, y_data, s=12, c="white", edgecolors="black", linewidths=0.4, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13)
    plt.show()
