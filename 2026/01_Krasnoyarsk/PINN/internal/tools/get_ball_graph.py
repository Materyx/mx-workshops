from __future__ import annotations

import sys
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERNAL = ROOT / "internal"
if str(INTERNAL) not in sys.path:
    sys.path.insert(0, str(INTERNAL))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.collections import LineCollection
from PIL import Image

from tools.get_ball_params import DEFAULT_DATA_CSV, get_ball_params
from tools.load_ball_track import load_ball_track

PINN_DIR = ROOT
ASSETS = ROOT / "assets"
TRACKS = ROOT / "internal" / "video" / "tracks"
FIGS_DIR = ASSETS / "ball_trajectory"

DEFAULT_BAD_DATA_CSV = TRACKS / "ball_throws" / "bad" / "track06.csv"

EXPERIMENTAL_COLOR = "#7B2CBF"
PREDICTION_COLOR = "#FF7F0E"
PREDICTION_COLOR_2 = "#0aaaaa"

BAD_DATA_ERROR_POINT_INDEX = 3
BAD_DATA_ERROR_POINT_Y = 0.621
BAD_DATA_TRAIN_EXCLUDE_TAIL = 2
BAD_DATA_TRAIN_EXCLUDE_INDICES = (BAD_DATA_ERROR_POINT_INDEX,)
THEORY_BALL_TRAIN_GIF_DURATION_MS = 10_000
THEORY_BALL_TRAIN_GIF_FINAL_PAUSE_MS = 5_000

# Как в pinn_ball_trajectory.py (явного seed в скрипте нет)
PINN_BALL_NUM_EPOCHS = 4000
PINN_BALL_LR = 0.01
PINN_BALL_N_HIDDEN = 20
PINN_BALL_PLOT_POINTS = 100
PINN_BALL_TRAIN_SEED = 42
PINN_BALL_TRAIN_SEED_PARTIAL = 43

# Как в pinn_ball_trajectory.py
PINN_BALL_ANALYTICAL_G = 9.8
PINN_BALL_ANALYTICAL_Y0 = 0.294
PINN_BALL_ANALYTICAL_V0 = 2.373
PINN_BALL_ANALYTICAL_N_POINTS = 100

PINN_LAMBDA_GIF_COLOR_07 = "#7B2CBF"
PINN_LAMBDA_GIF_COLOR_08 = "#0aaaaa"
PINN_LAMBDA_GIF_COLOR_09 = "#ff681d"
PINN_LAMBDA_GIF_COLOR_10 = "#228B22"
PINN_LAMBDA_GIF_DATA_COLOR = "black"

PINN_GRAD_BALANCE_LAMBDA_MIN = 1e-4
PINN_GRAD_BALANCE_LAMBDA_MAX = 100.0
PINN_GRAD_BALANCE_EMA = 0.9
PINN_GRAD_BALANCE_EPS = 1e-8
PINN_GRAD_BALANCE_EVERY_EPOCHS = 10

PINN_COMPARISON_CASES = (
    ("λ = (1, 0, 0)", 1.0, 0.0, 0.0, PINN_LAMBDA_GIF_COLOR_07),
    ("λ = (1, 1, 0)", 1.0, 1.0, 0.0, PINN_LAMBDA_GIF_COLOR_08),
    ("λ = (1, 1, 1)", 1.0, 1.0, 1.0, PINN_LAMBDA_GIF_COLOR_09),
)
PINN_COMPARISON_GRAD_BALANCE_LABEL = "Grad. balancing"


class BallMLP(nn.Module):
    """MLP y(t) — та же архитектура, что PINN в pinn_ball_trajectory.py."""

    def __init__(self, n_hidden: int = PINN_BALL_N_HIDDEN):
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


def _set_train_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _create_ball_mlp(n_hidden: int = PINN_BALL_N_HIDDEN, seed: int = PINN_BALL_TRAIN_SEED) -> BallMLP:
    _set_train_seed(seed)
    return BallMLP(n_hidden=n_hidden)


def _train_mse_only(
    tau: np.ndarray,
    y: np.ndarray,
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    train_seed: int = PINN_BALL_TRAIN_SEED,
) -> BallMLP:
    tau_tensor = torch.tensor(tau, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = _create_ball_mlp(n_hidden=n_hidden, seed=train_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(tau_tensor) - y_tensor) ** 2)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _predict_mlp(model: BallMLP, tau: np.ndarray) -> np.ndarray:
    tau_tensor = torch.tensor(tau, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        return model(tau_tensor).numpy().flatten()


def _apply_ball_plot_style(ax: plt.Axes) -> None:
    ax.set_xlim(-0.05, 0.65)
    ax.set_ylim(0.0, 0.8)
    ax.set_xticks(np.arange(0.0, 0.7, 0.1))
    ax.set_yticks(np.arange(0.0, 0.9, 0.1))
    ax.set_xlabel("t, с")
    ax.set_ylabel("y, м")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.set_autoscale_on(False)


def _apply_bad_data_point_error(
    y: np.ndarray,
    point_index: int = BAD_DATA_ERROR_POINT_INDEX,
    y_value: float = BAD_DATA_ERROR_POINT_Y,
) -> np.ndarray:
    y = y.copy()
    if point_index < 0 or point_index >= len(y):
        raise ValueError(
            f"Индекс ошибочной точки {point_index} вне диапазона [0, {len(y) - 1}]"
        )
    y[point_index] = y_value
    return y


def _load_bad_theory_points(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    exclude_tail: int = 0,
    *,
    apply_point_error: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    tau, y = _load_experimental_points(csv_path, exclude_tail=exclude_tail)
    if apply_point_error:
        y = _apply_bad_data_point_error(y)
    return tau, y


def _load_pinn_ball_track(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
) -> tuple[np.ndarray, np.ndarray]:
    """Исходные точки так же, как в pinn_ball_trajectory.py."""
    return load_ball_track(csv_path)


def _pinn_analytical_solution(tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    return (
        PINN_BALL_ANALYTICAL_Y0
        + PINN_BALL_ANALYTICAL_V0 * tau
        - 0.5 * PINN_BALL_ANALYTICAL_G * tau**2
    )


def _pinn_derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def _pinn_data_loss(model: BallMLP, tau: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((model(tau) - y) ** 2)


def _pinn_physics_loss(model: BallMLP, tau: torch.Tensor) -> torch.Tensor:
    tau = tau.clone().detach().requires_grad_(True)
    h_pred = model(tau)
    dh_dt_pred = _pinn_derivative(h_pred, tau)
    dh_dt_true = PINN_BALL_ANALYTICAL_V0 - PINN_BALL_ANALYTICAL_G * tau
    return torch.mean((dh_dt_pred - dh_dt_true) ** 2)


def _pinn_ic_loss(model: BallMLP) -> torch.Tensor:
    t0 = torch.zeros(1, 1, dtype=torch.float32)
    return (model(t0) - PINN_BALL_ANALYTICAL_Y0).pow(2).mean()


def _clip_pinn_lambda(value: float) -> float:
    return float(np.clip(value, PINN_GRAD_BALANCE_LAMBDA_MIN, PINN_GRAD_BALANCE_LAMBDA_MAX))


def _pinn_loss_grad_norm(loss: torch.Tensor, model: BallMLP) -> float:
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    sq_sum = sum(g.pow(2).sum() for g in grads if g is not None)
    if sq_sum == 0:
        return 0.0
    return float(torch.sqrt(sq_sum).item())


def _balance_pinn_lambdas_by_grad_norm(
    grad_norm_data: float,
    grad_norm_ode: float,
    grad_norm_ic: float,
    lambdas: tuple[float, float, float],
) -> tuple[float, float, float]:
    g_mean = (grad_norm_data + grad_norm_ode + grad_norm_ic) / 3.0
    new_lambdas = (
        _clip_pinn_lambda(g_mean / (grad_norm_data + PINN_GRAD_BALANCE_EPS)),
        _clip_pinn_lambda(g_mean / (grad_norm_ode + PINN_GRAD_BALANCE_EPS)),
        _clip_pinn_lambda(g_mean / (grad_norm_ic + PINN_GRAD_BALANCE_EPS)),
    )
    alpha = PINN_GRAD_BALANCE_EMA
    return tuple(
        alpha * old + (1.0 - alpha) * new for old, new in zip(lambdas, new_lambdas, strict=True)
    )


def _format_pinn_lambda_title(
    epoch: int,
    n_epochs: int,
    lambda_data: float,
    lambda_ode: float,
    lambda_ic: float,
) -> str:
    return (
        f"Эпоха {epoch}/{n_epochs} | "
        f"λ_data={lambda_data:g}, λ_ODE={lambda_ode:g}, λ_IC={lambda_ic:g}"
    )


def _load_pinn_theory_points(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
) -> tuple[np.ndarray, np.ndarray]:
    tau, y = _load_pinn_ball_track(csv_path)
    y = _apply_bad_data_point_error(y)
    return tau, y


def _analytical_plot_lines(
    tau_max: float,
    n_plot_points: int = PINN_BALL_ANALYTICAL_N_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    tau_line = np.linspace(0.0, tau_max, n_plot_points, dtype=np.float64)
    y_line = _pinn_analytical_solution(tau_line)
    valid = y_line >= 0.0
    return tau_line[valid], y_line[valid]


def _load_experimental_points(
    csv_path: Path | str,
    exclude_tail: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    params = get_ball_params(csv_path)
    if exclude_tail < 0:
        raise ValueError(f"exclude_tail должно быть >= 0, получено {exclude_tail}")
    if len(params.tau_all) <= exclude_tail:
        raise ValueError(
            f"После исключения {exclude_tail} точек не остаётся данных для графика"
        )
    tau = params.tau_all[:-exclude_tail] if exclude_tail else params.tau_all
    y = params.y_all[:-exclude_tail] if exclude_tail else params.y_all
    return tau, y


def _random_zero_mean_predictions(
    y: np.ndarray,
    seed: int = 13,
    error_scale: float = 0.1,
) -> np.ndarray:
    """
    Случайные предсказания вокруг y с нулевой средней ошибкой:
    (1/N) * sum(y_i - y_pred_i) = 0.
    """
    rng = np.random.default_rng(seed)
    errors = rng.normal(0.0, error_scale, size=y.shape)
    errors -= errors.mean()
    return y - errors


def _mean_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y - y_pred))


def _draw_ball_mse_axes(
    ax: plt.Axes,
    tau: np.ndarray,
    y: np.ndarray,
    tau_line: np.ndarray,
    y_line: np.ndarray,
    y_pred: np.ndarray,
    epoch: int | None = None,
    n_epochs: int | None = None,
) -> None:
    ax.clear()
    ax.scatter(tau, y, color=EXPERIMENTAL_COLOR, label="Зашумлённые данные")
    ax.plot(tau_line, y_line, color=PREDICTION_COLOR, label="Предикция нейронной сети")
    ax.scatter(tau, y_pred, color=PREDICTION_COLOR, s=35, zorder=3)
    _apply_ball_plot_style(ax)
    if epoch is not None and n_epochs is not None:
        ax.set_title(f"Эпоха {epoch}/{n_epochs}")


def _figure_to_image(fig: plt.Figure, dpi: int = 100, *, tight: bool = True) -> Image.Image:
    buffer = BytesIO()
    save_kwargs = {"format": "png", "dpi": dpi}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(buffer, **save_kwargs)
    buffer.seek(0)
    return Image.open(buffer).copy()


def _error_segments(
    tau: np.ndarray,
    y_exp: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            np.column_stack([tau, y_exp]),
            np.column_stack([tau, y_pred]),
        ],
        axis=1,
    )


def _train_mse_gif_frames(
    tau: np.ndarray,
    y: np.ndarray,
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    include_initial_frame: bool = True,
    frame_every_epochs: int = 10,
    gif_dpi: int = 100,
    prediction_label: str = "Предикция нейронной сети",
    train_seed: int = PINN_BALL_TRAIN_SEED,
) -> list[Image.Image]:
    if frame_every_epochs < 1:
        raise ValueError(f"frame_every_epochs должно быть >= 1, получено {frame_every_epochs}")

    tau_tensor = torch.tensor(tau, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    tau_line = np.linspace(float(tau[0]), float(tau[-1]), n_plot_points, dtype=np.float32)

    model = _create_ball_mlp(n_hidden=n_hidden, seed=train_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)

    ax.scatter(tau, y, color=EXPERIMENTAL_COLOR, label="Зашумлённые данные", zorder=4)
    (pred_line,) = ax.plot(
        tau_line,
        _predict_mlp(model, tau_line),
        color=PREDICTION_COLOR,
        label=prediction_label,
        zorder=3,
    )
    pred_scatter = ax.scatter(
        tau,
        _predict_mlp(model, tau),
        color=PREDICTION_COLOR,
        s=35,
        zorder=4,
    )
    _apply_ball_plot_style(ax)

    frames: list[Image.Image] = []

    def capture(epoch_label: int) -> None:
        model.eval()
        y_line = _predict_mlp(model, tau_line)
        y_pred = _predict_mlp(model, tau)
        pred_line.set_data(tau_line, y_line)
        pred_scatter.set_offsets(np.column_stack([tau, y_pred]))
        ax.set_title(f"Эпоха {epoch_label}/{n_epochs}")
        frames.append(_figure_to_image(fig, dpi=gif_dpi, tight=False))

    def should_capture(epoch: int) -> bool:
        return epoch % frame_every_epochs == 0 or epoch == n_epochs

    if include_initial_frame:
        capture(0)

    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss = torch.mean((model(tau_tensor) - y_tensor) ** 2)
        loss.backward()
        optimizer.step()
        if should_capture(epoch):
            capture(epoch)

    plt.close(fig)
    return frames


def _train_dual_mse_gif_frames(
    tau: np.ndarray,
    y: np.ndarray,
    train_exclude_tail: int = BAD_DATA_TRAIN_EXCLUDE_TAIL,
    train_exclude_indices: tuple[int, ...] = BAD_DATA_TRAIN_EXCLUDE_INDICES,
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    include_initial_frame: bool = True,
    frame_every_epochs: int = 10,
    gif_dpi: int = 100,
    train_seed: int = PINN_BALL_TRAIN_SEED,
    partial_train_seed: int = PINN_BALL_TRAIN_SEED_PARTIAL,
) -> list[Image.Image]:
    if frame_every_epochs < 1:
        raise ValueError(f"frame_every_epochs должно быть >= 1, получено {frame_every_epochs}")

    tau_tensor = torch.tensor(tau, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    train_indices = _partial_train_indices(
        len(tau),
        exclude_tail=train_exclude_tail,
        exclude_indices=train_exclude_indices,
    )
    tau_train = tau[train_indices]
    y_train = y[train_indices]
    tau_train_tensor = torch.tensor(tau_train, dtype=torch.float32).view(-1, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    tau_line = np.linspace(float(tau[0]), float(tau[-1]), n_plot_points, dtype=np.float32)

    model_all = _create_ball_mlp(n_hidden=n_hidden, seed=train_seed)
    model_partial = _create_ball_mlp(n_hidden=n_hidden, seed=partial_train_seed)
    optimizer_all = torch.optim.Adam(model_all.parameters(), lr=lr)
    optimizer_partial = torch.optim.Adam(model_partial.parameters(), lr=lr)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)

    ax.scatter(tau, y, color=EXPERIMENTAL_COLOR, label="Зашумлённые данные", zorder=4)
    (pred_line_1,) = ax.plot(
        tau_line,
        _predict_mlp(model_all, tau_line),
        color=PREDICTION_COLOR,
        label="Предикция нейронной сети 1",
        zorder=3,
    )
    pred_scatter_1 = ax.scatter(
        tau,
        _predict_mlp(model_all, tau),
        color=PREDICTION_COLOR,
        s=35,
        zorder=4,
    )
    (pred_line_2,) = ax.plot(
        tau_line,
        _predict_mlp(model_partial, tau_line),
        color=PREDICTION_COLOR_2,
        label="Предикция нейронной сети 2",
        zorder=3,
    )
    pred_scatter_2 = ax.scatter(
        tau,
        _predict_mlp(model_partial, tau),
        color=PREDICTION_COLOR_2,
        s=35,
        zorder=4,
    )
    _apply_ball_plot_style(ax)

    frames: list[Image.Image] = []

    def capture(epoch_label: int) -> None:
        model_all.eval()
        model_partial.eval()
        y_line_1 = _predict_mlp(model_all, tau_line)
        y_pred_1 = _predict_mlp(model_all, tau)
        y_line_2 = _predict_mlp(model_partial, tau_line)
        y_pred_2 = _predict_mlp(model_partial, tau)
        pred_line_1.set_data(tau_line, y_line_1)
        pred_scatter_1.set_offsets(np.column_stack([tau, y_pred_1]))
        pred_line_2.set_data(tau_line, y_line_2)
        pred_scatter_2.set_offsets(np.column_stack([tau, y_pred_2]))
        ax.set_title(f"Эпоха {epoch_label}/{n_epochs}")
        frames.append(_figure_to_image(fig, dpi=gif_dpi, tight=False))

    def should_capture(epoch: int) -> bool:
        return epoch % frame_every_epochs == 0 or epoch == n_epochs

    if include_initial_frame:
        capture(0)

    for epoch in range(1, n_epochs + 1):
        model_all.train()
        optimizer_all.zero_grad()
        loss_all = torch.mean((model_all(tau_tensor) - y_tensor) ** 2)
        loss_all.backward()
        optimizer_all.step()

        model_partial.train()
        optimizer_partial.zero_grad()
        loss_partial = torch.mean((model_partial(tau_train_tensor) - y_train_tensor) ** 2)
        loss_partial.backward()
        optimizer_partial.step()

        if should_capture(epoch):
            capture(epoch)

    plt.close(fig)
    return frames


def _setup_pinn_gif_axes(
    tau: np.ndarray,
    y: np.ndarray,
    tau_analytical: np.ndarray,
    y_analytical: np.ndarray,
    tau_line: np.ndarray,
    model: BallMLP,
    prediction_color: str,
    prediction_label: str = "Предикция нейронной сети",
) -> tuple[plt.Figure, plt.Axes, object, object]:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)

    ax.scatter(tau, y, color=PINN_LAMBDA_GIF_DATA_COLOR, label="Зашумлённые данные", zorder=4)
    ax.plot(
        tau_analytical,
        y_analytical,
        color="black",
        linestyle="--",
        label="Точное решение",
        zorder=2,
    )
    (pred_line,) = ax.plot(
        tau_line,
        _predict_mlp(model, tau_line),
        color=prediction_color,
        label=prediction_label,
        zorder=3,
    )
    pred_scatter = ax.scatter(
        tau,
        _predict_mlp(model, tau),
        color=prediction_color,
        s=35,
        zorder=4,
    )
    _apply_ball_plot_style(ax)
    return fig, ax, pred_line, pred_scatter


def _train_pinn_fixed_lambdas(
    tau: np.ndarray,
    y: np.ndarray,
    *,
    lambda_data: float,
    lambda_ode: float,
    lambda_ic: float,
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    train_seed: int = PINN_BALL_TRAIN_SEED,
) -> BallMLP:
    tau_tensor = torch.tensor(tau, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = _create_ball_mlp(n_hidden=n_hidden, seed=train_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = (
            lambda_data * _pinn_data_loss(model, tau_tensor, y_tensor)
            + lambda_ode * _pinn_physics_loss(model, tau_tensor)
            + lambda_ic * _pinn_ic_loss(model)
        )
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _train_pinn_grad_balance(
    tau: np.ndarray,
    y: np.ndarray,
    *,
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    train_seed: int = PINN_BALL_TRAIN_SEED,
    balance_every_epochs: int = PINN_GRAD_BALANCE_EVERY_EPOCHS,
    initial_lambdas: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> BallMLP:
    tau_tensor = torch.tensor(tau, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = _create_ball_mlp(n_hidden=n_hidden, seed=train_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lambda_data, lambda_ode, lambda_ic = initial_lambdas

    model.train()
    for epoch in range(1, n_epochs + 1):
        if epoch % balance_every_epochs == 0:
            l_data = _pinn_data_loss(model, tau_tensor, y_tensor)
            l_ode = _pinn_physics_loss(model, tau_tensor)
            l_ic = _pinn_ic_loss(model)
            lambda_data, lambda_ode, lambda_ic = _balance_pinn_lambdas_by_grad_norm(
                _pinn_loss_grad_norm(l_data, model),
                _pinn_loss_grad_norm(l_ode, model),
                _pinn_loss_grad_norm(l_ic, model),
                (lambda_data, lambda_ode, lambda_ic),
            )

        l_data = _pinn_data_loss(model, tau_tensor, y_tensor)
        l_ode = _pinn_physics_loss(model, tau_tensor)
        l_ic = _pinn_ic_loss(model)
        loss = lambda_data * l_data + lambda_ode * l_ode + lambda_ic * l_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _train_pinn_gif_frames(
    tau: np.ndarray,
    y: np.ndarray,
    tau_analytical: np.ndarray,
    y_analytical: np.ndarray,
    *,
    lambda_data: float,
    lambda_ode: float,
    lambda_ic: float,
    prediction_color: str,
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    include_initial_frame: bool = True,
    frame_every_epochs: int = 10,
    gif_dpi: int = 100,
    train_seed: int = PINN_BALL_TRAIN_SEED,
    prediction_label: str = "Предикция нейронной сети",
) -> list[Image.Image]:
    if frame_every_epochs < 1:
        raise ValueError(f"frame_every_epochs должно быть >= 1, получено {frame_every_epochs}")

    tau_tensor = torch.tensor(tau, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    tau_line = np.linspace(float(tau[0]), float(tau[-1]), n_plot_points, dtype=np.float32)

    model = _create_ball_mlp(n_hidden=n_hidden, seed=train_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fig, ax, pred_line, pred_scatter = _setup_pinn_gif_axes(
        tau,
        y,
        tau_analytical,
        y_analytical,
        tau_line,
        model,
        prediction_color,
        prediction_label,
    )

    frames: list[Image.Image] = []

    def capture(epoch_label: int) -> None:
        model.eval()
        y_line = _predict_mlp(model, tau_line)
        y_pred = _predict_mlp(model, tau)
        pred_line.set_data(tau_line, y_line)
        pred_scatter.set_offsets(np.column_stack([tau, y_pred]))
        ax.set_title(
            _format_pinn_lambda_title(
                epoch_label,
                n_epochs,
                lambda_data,
                lambda_ode,
                lambda_ic,
            )
        )
        frames.append(_figure_to_image(fig, dpi=gif_dpi, tight=False))

    def should_capture(epoch: int) -> bool:
        return epoch % frame_every_epochs == 0 or epoch == n_epochs

    if include_initial_frame:
        capture(0)

    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss = (
            lambda_data * _pinn_data_loss(model, tau_tensor, y_tensor)
            + lambda_ode * _pinn_physics_loss(model, tau_tensor)
            + lambda_ic * _pinn_ic_loss(model)
        )
        loss.backward()
        optimizer.step()
        if should_capture(epoch):
            capture(epoch)

    plt.close(fig)
    return frames


def _train_pinn_grad_balance_gif_frames(
    tau: np.ndarray,
    y: np.ndarray,
    tau_analytical: np.ndarray,
    y_analytical: np.ndarray,
    *,
    prediction_color: str,
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    include_initial_frame: bool = True,
    frame_every_epochs: int = 10,
    balance_every_epochs: int = PINN_GRAD_BALANCE_EVERY_EPOCHS,
    gif_dpi: int = 100,
    train_seed: int = PINN_BALL_TRAIN_SEED,
    prediction_label: str = "Предикция нейронной сети",
    initial_lambdas: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[Image.Image]:
    if frame_every_epochs < 1:
        raise ValueError(f"frame_every_epochs должно быть >= 1, получено {frame_every_epochs}")
    if balance_every_epochs < 1:
        raise ValueError(f"balance_every_epochs должно быть >= 1, получено {balance_every_epochs}")

    tau_tensor = torch.tensor(tau, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    tau_line = np.linspace(float(tau[0]), float(tau[-1]), n_plot_points, dtype=np.float32)

    model = _create_ball_mlp(n_hidden=n_hidden, seed=train_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fig, ax, pred_line, pred_scatter = _setup_pinn_gif_axes(
        tau,
        y,
        tau_analytical,
        y_analytical,
        tau_line,
        model,
        prediction_color,
        prediction_label,
    )

    lambda_data, lambda_ode, lambda_ic = initial_lambdas
    frames: list[Image.Image] = []

    def capture(epoch_label: int) -> None:
        model.eval()
        y_line = _predict_mlp(model, tau_line)
        y_pred = _predict_mlp(model, tau)
        pred_line.set_data(tau_line, y_line)
        pred_scatter.set_offsets(np.column_stack([tau, y_pred]))
        ax.set_title(
            _format_pinn_lambda_title(
                epoch_label,
                n_epochs,
                lambda_data,
                lambda_ode,
                lambda_ic,
            )
        )
        frames.append(_figure_to_image(fig, dpi=gif_dpi, tight=False))

    def should_capture(epoch: int) -> bool:
        return epoch % frame_every_epochs == 0 or epoch == n_epochs

    def should_balance(epoch: int) -> bool:
        return epoch % balance_every_epochs == 0

    if include_initial_frame:
        capture(0)

    model.train()
    for epoch in range(1, n_epochs + 1):
        if should_balance(epoch):
            l_data = _pinn_data_loss(model, tau_tensor, y_tensor)
            l_ode = _pinn_physics_loss(model, tau_tensor)
            l_ic = _pinn_ic_loss(model)
            lambda_data, lambda_ode, lambda_ic = _balance_pinn_lambdas_by_grad_norm(
                _pinn_loss_grad_norm(l_data, model),
                _pinn_loss_grad_norm(l_ode, model),
                _pinn_loss_grad_norm(l_ic, model),
                (lambda_data, lambda_ode, lambda_ic),
            )

        l_data = _pinn_data_loss(model, tau_tensor, y_tensor)
        l_ode = _pinn_physics_loss(model, tau_tensor)
        l_ic = _pinn_ic_loss(model)
        loss = lambda_data * l_data + lambda_ode * l_ode + lambda_ic * l_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if should_capture(epoch):
            capture(epoch)

    plt.close(fig)
    return frames


def _gif_frame_duration_ms(
    n_frames: int,
    total_duration_ms: int = THEORY_BALL_TRAIN_GIF_DURATION_MS,
) -> int:
    if n_frames < 1:
        raise ValueError(f"Число кадров должно быть >= 1, получено {n_frames}")
    return max(1, round(total_duration_ms / n_frames))


def _save_gif(
    frames: list[Image.Image],
    output_path: Path,
    frame_duration_ms: int | None = None,
    *,
    total_duration_ms: int = THEORY_BALL_TRAIN_GIF_DURATION_MS,
    final_pause_ms: int = 0,
) -> int:
    if not frames:
        raise ValueError("Нет кадров для сохранения GIF")
    if frame_duration_ms is None:
        frame_duration_ms = _gif_frame_duration_ms(len(frames), total_duration_ms)
    durations = [frame_duration_ms] * len(frames)
    if final_pause_ms > 0:
        durations[-1] = final_pause_ms
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return frame_duration_ms


def _partial_train_indices(
    n_points: int,
    exclude_tail: int = BAD_DATA_TRAIN_EXCLUDE_TAIL,
    exclude_indices: tuple[int, ...] = BAD_DATA_TRAIN_EXCLUDE_INDICES,
) -> np.ndarray:
    if exclude_tail < 0 or exclude_tail >= n_points:
        raise ValueError(
            f"exclude_tail должно быть в [0, {n_points - 1}], получено {exclude_tail}"
        )
    n_train = n_points - exclude_tail if exclude_tail else n_points
    exclude_set = set(exclude_indices)
    indices = [i for i in range(n_train) if i not in exclude_set]
    if not indices:
        raise ValueError("После исключения точек не остаётся данных для обучения")
    return np.array(indices, dtype=int)


def _plot_ball_mse_figure(
    tau: np.ndarray,
    y: np.ndarray,
    tau_line: np.ndarray,
    y_line: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_ball_mse_axes(ax, tau, y, tau_line, y_line, y_pred)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_ball_data_only_figure(
    tau: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    show: bool,
    label: str = "Зашумлённые данные",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, y, color=EXPERIMENTAL_COLOR, label=label)
    _apply_ball_plot_style(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_ball_analytical_figure(
    tau: np.ndarray,
    y: np.ndarray,
    tau_line: np.ndarray,
    y_line: np.ndarray,
    output_path: Path,
    show: bool,
    *,
    data_label: str = "Зашумлённые данные",
    analytical_label: str = "Точное решение",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, y, color=EXPERIMENTAL_COLOR, label=data_label, zorder=4)
    ax.plot(
        tau_line,
        y_line,
        color="black",
        linestyle="--",
        label=analytical_label,
        zorder=3,
    )
    _apply_ball_plot_style(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _draw_ball_theory_axes(
    ax: plt.Axes,
    tau: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    *,
    show_error_lines: bool = False,
) -> None:
    ax.clear()
    if show_error_lines:
        for t_i, y_i, y_p in zip(tau, y, y_pred):
            ax.plot(
                [t_i, t_i],
                [y_i, y_p],
                color="black",
                linewidth=0.8,
                zorder=2,
            )
    ax.scatter(tau, y, color=EXPERIMENTAL_COLOR, label="Экспериментальные данные", zorder=4)
    ax.scatter(tau, y_pred, color=PREDICTION_COLOR, label="Предикция нейронной сети", zorder=4)
    _apply_ball_plot_style(ax)


def _smoothstep(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _compute_mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y - y_pred) ** 2))


def _format_mse_label(mse: float) -> str:
    text = f"{mse:.4f}".replace(".", ",")
    return f"MSE = {text}"


def _theory_prediction_transition_frames(
    tau: np.ndarray,
    y: np.ndarray,
    y_pred_start: np.ndarray,
    y_pred_end: np.ndarray,
    n_frames: int = 60,
    gif_dpi: int = 100,
) -> list[Image.Image]:
    if n_frames < 2:
        raise ValueError(f"n_frames должно быть >= 2, получено {n_frames}")

    alphas = _smoothstep(np.linspace(0.0, 1.0, n_frames))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)

    ax.scatter(tau, y, color=EXPERIMENTAL_COLOR, label="Экспериментальные данные", zorder=4)
    pred_scatter = ax.scatter(
        tau,
        y_pred_start,
        color=PREDICTION_COLOR,
        label="Предикция нейронной сети",
        zorder=4,
    )
    error_lines = LineCollection(
        _error_segments(tau, y, y_pred_start),
        colors="black",
        linewidths=0.8,
        zorder=2,
    )
    ax.add_collection(error_lines)
    _apply_ball_plot_style(ax)
    ax.set_title(_format_mse_label(_compute_mse(y, y_pred_start)))

    frames: list[Image.Image] = []
    for alpha in alphas:
        y_pred = (1.0 - alpha) * y_pred_start + alpha * y_pred_end
        pred_scatter.set_offsets(np.column_stack([tau, y_pred]))
        error_lines.set_segments(_error_segments(tau, y, y_pred))
        ax.set_title(_format_mse_label(_compute_mse(y, y_pred)))
        frames.append(_figure_to_image(fig, dpi=gif_dpi, tight=False))

    plt.close(fig)
    return frames


def _plot_ball_theory_figure(
    tau: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_ball_theory_axes(ax, tau, y, y_pred)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_theory_ball_01(
    csv_path: Path | str = DEFAULT_DATA_CSV,
    exclude_tail: int = 2,
    output_path: Path | str = FIGS_DIR / "theory_ball_01.png",
    prediction_seed: int = 13,
    error_scale: float = 0.1,
    show: bool = False,
) -> Path:
    """Экспериментальные точки и случайная «предикция» с нулевой средней ошибкой."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_experimental_points(csv_path, exclude_tail=exclude_tail)
    y_pred = _random_zero_mean_predictions(
        y,
        seed=prediction_seed,
        error_scale=error_scale,
    )

    _plot_ball_theory_figure(tau, y, y_pred, output_path, show)
    return output_path


def plot_theory_ball_02(
    csv_path: Path | str = DEFAULT_DATA_CSV,
    exclude_tail: int = 2,
    output_path: Path | str = FIGS_DIR / "theory_ball_02.png",
    prediction_seed: int = 13,
    error_scale: float = 0.008,
    show: bool = False,
) -> Path:
    """Почти совпадающая с данными «предикция» с небольшими отклонениями."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_experimental_points(csv_path, exclude_tail=exclude_tail)
    y_pred = _random_zero_mean_predictions(
        y,
        seed=prediction_seed,
        error_scale=error_scale,
    )

    _plot_ball_theory_figure(tau, y, y_pred, output_path, show)
    return output_path


THEORY_BALL_01_ERROR_SCALE = 0.1
THEORY_BALL_02_ERROR_SCALE = 0.008


def plot_theory_ball_02_gif(
    csv_path: Path | str = DEFAULT_DATA_CSV,
    exclude_tail: int = 2,
    output_path: Path | str = FIGS_DIR / "theory_ball_02.gif",
    prediction_seed: int = 13,
    error_scale_start: float = THEORY_BALL_01_ERROR_SCALE,
    error_scale_end: float = THEORY_BALL_02_ERROR_SCALE,
    n_frames: int = 60,
    frame_duration_ms: int = 80,
    show: bool = False,
) -> Path:
    """Плавный переход предикции из theory_ball_01 в theory_ball_02."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_experimental_points(csv_path, exclude_tail=exclude_tail)
    y_pred_start = _random_zero_mean_predictions(
        y,
        seed=prediction_seed,
        error_scale=error_scale_start,
    )
    y_pred_end = _random_zero_mean_predictions(
        y,
        seed=prediction_seed,
        error_scale=error_scale_end,
    )
    frames = _theory_prediction_transition_frames(
        tau,
        y,
        y_pred_start,
        y_pred_end,
        n_frames=n_frames,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()

    return output_path


def plot_theory_ball_03(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    exclude_tail: int = 0,
    output_path: Path | str = FIGS_DIR / "theory_ball_03.png",
    show: bool = False,
) -> Path:
    """Зашумлённые данные bad/track06 — только фиолетовые точки, все измерения."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_bad_theory_points(csv_path, exclude_tail=exclude_tail)
    _plot_ball_data_only_figure(tau, y, output_path, show)
    return output_path


def plot_theory_ball_04(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    exclude_tail: int = 0,
    output_path: Path | str = FIGS_DIR / "theory_ball_04.png",
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    show: bool = False,
) -> Path:
    """Зашумлённые данные + MLP, обученная только по MSE (без PINN)."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_bad_theory_points(csv_path, exclude_tail=exclude_tail)
    model = _train_mse_only(tau, y, n_epochs=n_epochs, lr=lr, n_hidden=n_hidden)

    tau_line = np.linspace(float(tau[0]), float(tau[-1]), n_plot_points, dtype=np.float32)
    y_line = _predict_mlp(model, tau_line)
    y_pred = _predict_mlp(model, tau)

    _plot_ball_mse_figure(tau, y, tau_line, y_line, y_pred, output_path, show)
    return output_path


def plot_theory_ball_04_gif(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    exclude_tail: int = 0,
    output_path: Path | str = FIGS_DIR / "theory_ball_04.gif",
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    frame_duration_ms: int | None = None,
    frame_every_epochs: int = 10,
    include_initial_frame: bool = True,
    show: bool = False,
) -> Path:
    """GIF обучения MLP по MSE: кадр каждые frame_every_epochs эпох."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_bad_theory_points(csv_path, exclude_tail=exclude_tail)
    frames = _train_mse_gif_frames(
        tau,
        y,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        n_plot_points=n_plot_points,
        include_initial_frame=include_initial_frame,
        frame_every_epochs=frame_every_epochs,
    )
    _save_gif(
        frames,
        output_path,
        frame_duration_ms=frame_duration_ms,
        final_pause_ms=THEORY_BALL_TRAIN_GIF_FINAL_PAUSE_MS,
    )

    if show:
        frames[0].show()

    return output_path


def plot_theory_ball_05_gif(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    exclude_tail: int = 0,
    output_path: Path | str = FIGS_DIR / "theory_ball_05.gif",
    train_exclude_tail: int = BAD_DATA_TRAIN_EXCLUDE_TAIL,
    train_exclude_indices: tuple[int, ...] = BAD_DATA_TRAIN_EXCLUDE_INDICES,
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    frame_duration_ms: int | None = None,
    frame_every_epochs: int = 10,
    include_initial_frame: bool = True,
    show: bool = False,
) -> Path:
    """GIF: MLP на всех данных + MLP без последних точек и без выброса."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_bad_theory_points(csv_path, exclude_tail=exclude_tail)
    frames = _train_dual_mse_gif_frames(
        tau,
        y,
        train_exclude_tail=train_exclude_tail,
        train_exclude_indices=train_exclude_indices,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        n_plot_points=n_plot_points,
        include_initial_frame=include_initial_frame,
        frame_every_epochs=frame_every_epochs,
    )
    _save_gif(
        frames,
        output_path,
        frame_duration_ms=frame_duration_ms,
        final_pause_ms=THEORY_BALL_TRAIN_GIF_FINAL_PAUSE_MS,
    )

    if show:
        frames[0].show()

    return output_path


def plot_theory_ball_06(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    output_path: Path | str = FIGS_DIR / "theory_ball_06.png",
    n_plot_points: int = PINN_BALL_ANALYTICAL_N_POINTS,
    show: bool = False,
) -> Path:
    """Исходные данные + штриховая аналитика из pinn_ball_trajectory.py."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_pinn_ball_track(csv_path)
    y = _apply_bad_data_point_error(y)
    tau_line = np.linspace(0.0, float(tau[-1]), n_plot_points, dtype=np.float64)
    y_line = _pinn_analytical_solution(tau_line)
    valid = y_line >= 0.0
    tau_line = tau_line[valid]
    y_line = y_line[valid]

    _plot_ball_analytical_figure(tau, y, tau_line, y_line, output_path, show)
    return output_path


def _plot_theory_ball_pinn_lambda_gif(
    *,
    output_path: Path | str,
    lambda_data: float,
    lambda_ode: float,
    lambda_ic: float,
    prediction_color: str,
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    frame_duration_ms: int | None = None,
    frame_every_epochs: int = 10,
    include_initial_frame: bool = True,
    train_seed: int = PINN_BALL_TRAIN_SEED,
    show: bool = False,
) -> Path:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_bad_theory_points(csv_path)
    tau_analytical, y_analytical = _analytical_plot_lines(float(tau[-1]), n_plot_points)
    frames = _train_pinn_gif_frames(
        tau,
        y,
        tau_analytical,
        y_analytical,
        lambda_data=lambda_data,
        lambda_ode=lambda_ode,
        lambda_ic=lambda_ic,
        prediction_color=prediction_color,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        n_plot_points=n_plot_points,
        include_initial_frame=include_initial_frame,
        frame_every_epochs=frame_every_epochs,
        train_seed=train_seed,
    )
    _save_gif(
        frames,
        output_path,
        frame_duration_ms=frame_duration_ms,
        final_pause_ms=THEORY_BALL_TRAIN_GIF_FINAL_PAUSE_MS,
    )

    if show:
        frames[0].show()

    return output_path


def plot_theory_ball_07_gif(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    output_path: Path | str = FIGS_DIR / "theory_ball_07.gif",
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    frame_duration_ms: int | None = None,
    frame_every_epochs: int = 10,
    include_initial_frame: bool = True,
    show: bool = False,
) -> Path:
    """GIF PINN: λ_data=1, λ_ODE=0, λ_IC=0."""
    return _plot_theory_ball_pinn_lambda_gif(
        output_path=output_path,
        csv_path=csv_path,
        lambda_data=1.0,
        lambda_ode=0.0,
        lambda_ic=0.0,
        prediction_color=PINN_LAMBDA_GIF_COLOR_07,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        n_plot_points=n_plot_points,
        frame_duration_ms=frame_duration_ms,
        frame_every_epochs=frame_every_epochs,
        include_initial_frame=include_initial_frame,
        show=show,
    )


def plot_theory_ball_08_gif(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    output_path: Path | str = FIGS_DIR / "theory_ball_08.gif",
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    frame_duration_ms: int | None = None,
    frame_every_epochs: int = 10,
    include_initial_frame: bool = True,
    show: bool = False,
) -> Path:
    """GIF PINN: λ_data=1, λ_ODE=1, λ_IC=0."""
    return _plot_theory_ball_pinn_lambda_gif(
        output_path=output_path,
        csv_path=csv_path,
        lambda_data=1.0,
        lambda_ode=1.0,
        lambda_ic=0.0,
        prediction_color=PINN_LAMBDA_GIF_COLOR_08,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        n_plot_points=n_plot_points,
        frame_duration_ms=frame_duration_ms,
        frame_every_epochs=frame_every_epochs,
        include_initial_frame=include_initial_frame,
        show=show,
    )


def plot_theory_ball_09_gif(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    output_path: Path | str = FIGS_DIR / "theory_ball_09.gif",
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    frame_duration_ms: int | None = None,
    frame_every_epochs: int = 10,
    include_initial_frame: bool = True,
    show: bool = False,
) -> Path:
    """GIF PINN: λ_data=1, λ_ODE=1, λ_IC=1."""
    return _plot_theory_ball_pinn_lambda_gif(
        output_path=output_path,
        csv_path=csv_path,
        lambda_data=1.0,
        lambda_ode=1.0,
        lambda_ic=1.0,
        prediction_color=PINN_LAMBDA_GIF_COLOR_09,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        n_plot_points=n_plot_points,
        frame_duration_ms=frame_duration_ms,
        frame_every_epochs=frame_every_epochs,
        include_initial_frame=include_initial_frame,
        show=show,
    )


def plot_theory_ball_10_gif(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    output_path: Path | str = FIGS_DIR / "theory_ball_10.gif",
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    frame_duration_ms: int | None = None,
    frame_every_epochs: int = 10,
    balance_every_epochs: int = PINN_GRAD_BALANCE_EVERY_EPOCHS,
    include_initial_frame: bool = True,
    show: bool = False,
) -> Path:
    """GIF PINN: gradient balancing λ (старт 1,1,1)."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_bad_theory_points(csv_path)
    tau_analytical, y_analytical = _analytical_plot_lines(float(tau[-1]), n_plot_points)
    frames = _train_pinn_grad_balance_gif_frames(
        tau,
        y,
        tau_analytical,
        y_analytical,
        prediction_color=PINN_LAMBDA_GIF_COLOR_10,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        n_plot_points=n_plot_points,
        include_initial_frame=include_initial_frame,
        frame_every_epochs=frame_every_epochs,
        balance_every_epochs=balance_every_epochs,
        train_seed=PINN_BALL_TRAIN_SEED,
    )
    _save_gif(
        frames,
        output_path,
        frame_duration_ms=frame_duration_ms,
        final_pause_ms=THEORY_BALL_TRAIN_GIF_FINAL_PAUSE_MS,
    )

    if show:
        frames[0].show()

    return output_path


def _plot_pinn_comparison_figure(
    tau: np.ndarray,
    y: np.ndarray,
    tau_analytical: np.ndarray,
    y_analytical: np.ndarray,
    models: list[tuple[str, BallMLP, str]],
    output_path: Path,
    show: bool,
    *,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
) -> None:
    tau_line = np.linspace(float(tau[0]), float(tau[-1]), n_plot_points, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, y, color=PINN_LAMBDA_GIF_DATA_COLOR, label="Зашумлённые данные", zorder=4)
    ax.plot(
        tau_analytical,
        y_analytical,
        color="black",
        linestyle="--",
        label="Точное решение",
        zorder=2,
    )
    for label, model, color in models:
        ax.plot(
            tau_line,
            _predict_mlp(model, tau_line),
            color=color,
            label=label,
            zorder=3,
        )

    _apply_ball_plot_style(ax)
    ax.set_title("Сравнение подходов")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_theory_ball_11(
    csv_path: Path | str = DEFAULT_BAD_DATA_CSV,
    output_path: Path | str = FIGS_DIR / "theory_ball_11.png",
    n_epochs: int = PINN_BALL_NUM_EPOCHS,
    lr: float = PINN_BALL_LR,
    n_hidden: int = PINN_BALL_N_HIDDEN,
    n_plot_points: int = PINN_BALL_PLOT_POINTS,
    balance_every_epochs: int = PINN_GRAD_BALANCE_EVERY_EPOCHS,
    show: bool = False,
) -> Path:
    """Сравнение финальных PINN: конфигурации 07–10 на одном графике."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    tau, y = _load_bad_theory_points(csv_path)
    tau_analytical, y_analytical = _analytical_plot_lines(float(tau[-1]), n_plot_points)

    models: list[tuple[str, BallMLP, str]] = []
    for label, lambda_data, lambda_ode, lambda_ic, color in PINN_COMPARISON_CASES:
        model = _train_pinn_fixed_lambdas(
            tau,
            y,
            lambda_data=lambda_data,
            lambda_ode=lambda_ode,
            lambda_ic=lambda_ic,
            n_epochs=n_epochs,
            lr=lr,
            n_hidden=n_hidden,
            train_seed=PINN_BALL_TRAIN_SEED,
        )
        models.append((label, model, color))

    grad_balance_model = _train_pinn_grad_balance(
        tau,
        y,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        train_seed=PINN_BALL_TRAIN_SEED,
        balance_every_epochs=balance_every_epochs,
    )
    models.append(
        (PINN_COMPARISON_GRAD_BALANCE_LABEL, grad_balance_model, PINN_LAMBDA_GIF_COLOR_10)
    )

    _plot_pinn_comparison_figure(
        tau,
        y,
        tau_analytical,
        y_analytical,
        models,
        output_path,
        show,
        n_plot_points=n_plot_points,
    )
    return output_path


PLOTS: dict[str, Callable[..., Path]] = {
    "theory_ball_01": plot_theory_ball_01,
    "theory_ball_02": plot_theory_ball_02,
    "theory_ball_02_gif": plot_theory_ball_02_gif,
    "theory_ball_03": plot_theory_ball_03,
    "theory_ball_04": plot_theory_ball_04,
    "theory_ball_04_gif": plot_theory_ball_04_gif,
    "theory_ball_05_gif": plot_theory_ball_05_gif,
    "theory_ball_06": plot_theory_ball_06,
    "theory_ball_07_gif": plot_theory_ball_07_gif,
    "theory_ball_08_gif": plot_theory_ball_08_gif,
    "theory_ball_09_gif": plot_theory_ball_09_gif,
    "theory_ball_10_gif": plot_theory_ball_10_gif,
    "theory_ball_11": plot_theory_ball_11,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Генерация учебных графиков траектории мяча")
    parser.add_argument(
        "plot",
        nargs="?",
        default="theory_ball_01",
        choices=sorted(PLOTS),
        help="Имя рисунка для генерации",
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument("--exclude-tail", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prediction-seed", type=int, default=13)
    parser.add_argument("--error-scale", type=float, default=None)
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=None,
        help="Длительность кадра в мс; по умолчанию GIF ~10 с",
    )
    parser.add_argument("--frame-every-epochs", type=int, default=10)
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    plot_fn = PLOTS[args.plot]
    kwargs: dict = {
        "show": args.show,
    }
    if args.csv is not None:
        kwargs["csv_path"] = args.csv
    if args.plot in {"theory_ball_01", "theory_ball_02", "theory_ball_02_gif"}:
        kwargs["exclude_tail"] = args.exclude_tail
        kwargs["prediction_seed"] = args.prediction_seed
        if args.error_scale is not None:
            kwargs["error_scale"] = args.error_scale
    if args.plot == "theory_ball_02_gif":
        kwargs["n_frames"] = args.n_frames
        kwargs["frame_duration_ms"] = args.frame_duration_ms
    if args.plot in {
        "theory_ball_04",
        "theory_ball_04_gif",
        "theory_ball_05_gif",
        "theory_ball_07_gif",
        "theory_ball_08_gif",
        "theory_ball_09_gif",
        "theory_ball_10_gif",
        "theory_ball_11",
    } and args.train_epochs is not None:
        kwargs["n_epochs"] = args.train_epochs
    if args.plot in {
        "theory_ball_04_gif",
        "theory_ball_05_gif",
        "theory_ball_07_gif",
        "theory_ball_08_gif",
        "theory_ball_09_gif",
        "theory_ball_10_gif",
    }:
        if args.frame_duration_ms is not None:
            kwargs["frame_duration_ms"] = args.frame_duration_ms
        kwargs["frame_every_epochs"] = args.frame_every_epochs
    if args.output is not None:
        kwargs["output_path"] = args.output

    result = plot_fn(**kwargs)
    print(f"Сохранено: {result}")
    if args.plot == "theory_ball_02_gif":
        print(f"Кадров: {args.n_frames} (theory_ball_01 → theory_ball_02)")
    if args.plot in {
        "theory_ball_04_gif",
        "theory_ball_05_gif",
        "theory_ball_07_gif",
        "theory_ball_08_gif",
        "theory_ball_09_gif",
        "theory_ball_10_gif",
    }:
        n_epochs = args.train_epochs if args.train_epochs is not None else PINN_BALL_NUM_EPOCHS
        n_frames = n_epochs // args.frame_every_epochs
        if n_epochs % args.frame_every_epochs != 0:
            n_frames += 1
        n_frames += 1  # эпоха 0
        frame_duration_ms = args.frame_duration_ms
        if frame_duration_ms is None:
            frame_duration_ms = _gif_frame_duration_ms(n_frames)
        total_duration_s = (
            (n_frames - 1) * frame_duration_ms + THEORY_BALL_TRAIN_GIF_FINAL_PAUSE_MS
        ) / 1000
        print(
            f"Кадров: {n_frames} "
            f"(эпоха 0 + каждые {args.frame_every_epochs} эпох до {n_epochs}), "
            f"{frame_duration_ms} мс/кадр, пауза на финале "
            f"{THEORY_BALL_TRAIN_GIF_FINAL_PAUSE_MS} мс, ~{total_duration_s:.1f} с"
        )

    if args.plot in {"theory_ball_01", "theory_ball_02"}:
        csv_path = args.csv if args.csv is not None else DEFAULT_DATA_CSV
        _, y = _load_experimental_points(csv_path, exclude_tail=args.exclude_tail)
        error_scale = args.error_scale
        if error_scale is None:
            error_scale = 0.008 if args.plot == "theory_ball_02" else 0.1
        y_pred = _random_zero_mean_predictions(
            y,
            seed=args.prediction_seed,
            error_scale=error_scale,
        )
        mean_error = _mean_error(y, y_pred)
        print(f"Средняя ошибка (1/N)*sum(y - y_pred) = {mean_error:.6e}")
