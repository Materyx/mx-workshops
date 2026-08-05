from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
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
from PIL import Image

from tools.get_pendulum_params import PendulumParams, get_pendulum_params, sample_pendulum_data

PINN_DIR = ROOT
ASSETS = ROOT / "assets"
TRACKS = ROOT / "internal" / "video" / "tracks"
FIGS_DIR = ASSETS / "pendulum"
DEFAULT_DATA_CSV = TRACKS / "pendulums" / "pendulum_2_trajectory.csv"

PENDULUM_T_MIN = 0.0
PENDULUM_T_MAX = 60.0
PENDULUM_N_DATA = 50
PENDULUM_DATA_SEED = 42
PENDULUM_TRAIN_SEED = 42
PENDULUM_N_HIDDEN = 60
PENDULUM_LR = 0.01
PENDULUM_NUM_EPOCHS = 10_000
PENDULUM_N_PHYS = 500
PENDULUM_PLOT_POINTS_PER_PERIOD = 40
PENDULUM_USE_FOURIER = False
PENDULUM_PINN_USE_FOURIER = True

PENDULUM_LAMBDA_DATA = 48.0
PENDULUM_LAMBDA_ODE = 0.39
PENDULUM_LAMBDA_IC = 0.23
PENDULUM_LAMBDA_IC_VEL = 0.35

PENDULUM_GIF_DURATION_MS = 10_000
PENDULUM_GIF_FINAL_PAUSE_MS = 5_000
PENDULUM_GIF_FRAME_EVERY_EPOCHS = 25
PENDULUM_GIF_DPI = 100
PENDULUM_FIG_WIDTH = 10.0
PENDULUM_FIG_HEIGHT = 5.0

PENDULUM_DATA_COLOR = "black"
PENDULUM_VANILLA_COLOR = "#7B2CBF"
PENDULUM_PINN_PLAIN_COLOR = "#0aaaaa"
PENDULUM_PINN_FOURIER_COLOR = "#FF7F0E"
PENDULUM_PREDICTION_COLOR = "blue"


class PendulumMLP(nn.Module):
    """MLP θ(t) — архитектура из pinn_pendulum.py."""

    def __init__(
        self,
        n_hidden: int = PENDULUM_N_HIDDEN,
        omega_feature: float = 1.0,
        use_fourier_features: bool = PENDULUM_USE_FOURIER,
    ):
        super().__init__()
        self.use_fourier_features = use_fourier_features
        self.omega_feature = float(omega_feature)
        input_dim = 3 if use_fourier_features else 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def features(self, t: torch.Tensor) -> torch.Tensor:
        if self.use_fourier_features:
            omega_t = self.omega_feature * t
            return torch.cat([t, torch.sin(omega_t), torch.cos(omega_t)], dim=1)
        return t

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(t))


def _set_train_seed(seed: int = PENDULUM_TRAIN_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _create_pendulum_mlp(
    params: PendulumParams,
    n_hidden: int = PENDULUM_N_HIDDEN,
    seed: int = PENDULUM_TRAIN_SEED,
    *,
    use_fourier_features: bool = PENDULUM_USE_FOURIER,
) -> PendulumMLP:
    _set_train_seed(seed)
    return PendulumMLP(
        n_hidden=n_hidden,
        omega_feature=params.pinn_omega,
        use_fourier_features=use_fourier_features,
    )


def _derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]


def _physics_loss(model: PendulumMLP, t_phys: torch.Tensor, params: PendulumParams) -> torch.Tensor:
    t_grad = t_phys.clone().detach().requires_grad_(True)
    theta_pred = model(t_grad)
    theta_t = _derivative(theta_pred, t_grad)
    theta_tt = _derivative(theta_t, t_grad)
    residual = (
        theta_tt
        + 2.0 * params.pinn_beta_phys * theta_t
        + params.pinn_omega0_sq * theta_pred
    )
    return torch.mean(residual ** 2)


def _initial_condition_loss(model: PendulumMLP, params: PendulumParams) -> torch.Tensor:
    t0 = torch.zeros(1, 1, dtype=torch.float32)
    theta_pred_0 = model(t0)
    return (theta_pred_0 - params.pinn_theta0).pow(2).mean()


def _initial_velocity_loss(model: PendulumMLP, params: PendulumParams) -> torch.Tensor:
    t0 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True)
    theta_pred_0 = model(t0)
    theta_t_0 = _derivative(theta_pred_0, t0)
    return (theta_t_0 - params.pinn_theta_dot0).pow(2).mean()


def _data_loss(
    model: PendulumMLP,
    t_data: torch.Tensor,
    theta_data: torch.Tensor,
) -> torch.Tensor:
    theta_pred = model(t_data)
    return torch.mean((theta_pred - theta_data) ** 2)


def _predict_mlp(model: PendulumMLP, t_values: np.ndarray) -> np.ndarray:
    t_tensor = torch.tensor(t_values, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        return model(t_tensor).numpy().flatten()


def _evaluation_grid(params: PendulumParams) -> np.ndarray:
    n_points = max(
        400,
        int(
            (params.t_max - params.t_min)
            * params.pinn_omega
            / (2.0 * np.pi)
            * PENDULUM_PLOT_POINTS_PER_PERIOD
        ),
    )
    return np.linspace(params.t_min, params.t_max, n_points, dtype=np.float32)


def _analytical_plot_lines(params: PendulumParams) -> tuple[np.ndarray, np.ndarray]:
    n_points = max(
        200,
        int(
            (params.t_max - params.t_min)
            * params.omega0
            / (2.0 * np.pi)
            * PENDULUM_PLOT_POINTS_PER_PERIOD
        ),
    )
    t_line = np.linspace(params.t_min, params.t_max, n_points, dtype=np.float32)
    theta_line = params.evaluate(t_line)
    return t_line, theta_line.astype(np.float32)


def _plot_y_limits(theta_analytical: np.ndarray, margin: float = 0.08) -> tuple[float, float]:
    amplitude = float(np.max(np.abs(theta_analytical)))
    limit = max(amplitude * (1.0 + margin), 0.25)
    return -limit, limit


def _apply_pendulum_plot_style(
    ax: plt.Axes,
    t_min: float,
    t_max: float,
    y_min: float,
    y_max: float,
) -> None:
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("t")
    ax.set_ylabel("θ(t) [rad]")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_autoscale_on(False)


def _format_pinn_lambda_title(
    epoch: int,
    n_epochs: int,
    lambda_data: float,
    lambda_ode: float,
    lambda_ic: float,
    lambda_ic_vel: float,
) -> str:
    return (
        f"Эпоха {epoch}/{n_epochs} | "
        f"λ_data={lambda_data:g}, λ_ode={lambda_ode:g}, "
        f"λ_ic={lambda_ic:g}, λ_ic,vel={lambda_ic_vel:g}"
    )


def _figure_to_image(fig: plt.Figure, dpi: int = PENDULUM_GIF_DPI) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    buffer.seek(0)
    return Image.open(buffer).copy()


def _gif_frame_duration_ms(
    n_frames: int,
    total_duration_ms: int = PENDULUM_GIF_DURATION_MS,
) -> int:
    if n_frames < 1:
        raise ValueError(f"Число кадров должно быть >= 1, получено {n_frames}")
    return max(1, round(total_duration_ms / n_frames))


def _save_gif(
    frames: list[Image.Image],
    output_path: Path,
    frame_duration_ms: int | None = None,
    *,
    total_duration_ms: int = PENDULUM_GIF_DURATION_MS,
    final_pause_ms: int = PENDULUM_GIF_FINAL_PAUSE_MS,
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


def _setup_pendulum_gif_axes(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    t_analytical: np.ndarray,
    theta_analytical: np.ndarray,
    t_line: np.ndarray,
    model: PendulumMLP,
    params: PendulumParams,
    *,
    title: str,
    prediction_label: str = "Предсказание модели",
    prediction_color: str = PENDULUM_PREDICTION_COLOR,
) -> tuple[plt.Figure, plt.Axes, object]:
    y_min, y_max = _plot_y_limits(theta_analytical)

    fig, ax = plt.subplots(figsize=(PENDULUM_FIG_WIDTH, PENDULUM_FIG_HEIGHT))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.12)

    ax.scatter(
        t_data,
        theta_data,
        color=PENDULUM_DATA_COLOR,
        s=18,
        label="Данные (выборка)",
        zorder=4,
    )
    ax.plot(
        t_analytical,
        theta_analytical,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Аналитическое решение",
        zorder=2,
    )
    (pred_line,) = ax.plot(
        t_line,
        _predict_mlp(model, t_line),
        color=prediction_color,
        linewidth=1.2,
        label=prediction_label,
        zorder=3,
    )
    _apply_pendulum_plot_style(ax, params.t_min, params.t_max, y_min, y_max)
    ax.set_title(title)
    return fig, ax, pred_line


def _update_prediction_line(
    model: PendulumMLP,
    t_line: np.ndarray,
    pred_line: object,
) -> None:
    pred_line.set_data(t_line, _predict_mlp(model, t_line))


def _train_vanilla_gif_frames(
    params: PendulumParams,
    t_data: np.ndarray,
    theta_data: np.ndarray,
    t_analytical: np.ndarray,
    theta_analytical: np.ndarray,
    *,
    n_epochs: int = PENDULUM_NUM_EPOCHS,
    lr: float = PENDULUM_LR,
    n_hidden: int = PENDULUM_N_HIDDEN,
    include_initial_frame: bool = True,
    frame_every_epochs: int = PENDULUM_GIF_FRAME_EVERY_EPOCHS,
    gif_dpi: int = PENDULUM_GIF_DPI,
    train_seed: int = PENDULUM_TRAIN_SEED,
) -> list[Image.Image]:
    if frame_every_epochs < 1:
        raise ValueError(f"frame_every_epochs должно быть >= 1, получено {frame_every_epochs}")

    t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
    theta_data_tensor = torch.tensor(theta_data, dtype=torch.float32).view(-1, 1)
    t_line = _evaluation_grid(params)

    model = _create_pendulum_mlp(params, n_hidden=n_hidden, seed=train_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fig, ax, pred_line = _setup_pendulum_gif_axes(
        t_data,
        theta_data,
        t_analytical,
        theta_analytical,
        t_line,
        model,
        params,
        title="Эпоха 0/{} | Базовая модель".format(n_epochs),
        prediction_label="Базовая модель",
        prediction_color=PENDULUM_VANILLA_COLOR,
    )

    frames: list[Image.Image] = []

    def capture(epoch_label: int) -> None:
        model.eval()
        _update_prediction_line(model, t_line, pred_line)
        ax.set_title(f"Эпоха {epoch_label}/{n_epochs} | Базовая модель")
        frames.append(_figure_to_image(fig, dpi=gif_dpi))
        model.train()

    def should_capture(epoch: int) -> bool:
        return epoch % frame_every_epochs == 0 or epoch == n_epochs - 1

    if include_initial_frame:
        capture(0)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = _data_loss(model, t_data_tensor, theta_data_tensor)
        loss.backward()
        optimizer.step()
        if should_capture(epoch):
            capture(epoch + 1)

    plt.close(fig)
    return frames


def _train_pinn_gif_frames(
    params: PendulumParams,
    t_data: np.ndarray,
    theta_data: np.ndarray,
    t_analytical: np.ndarray,
    theta_analytical: np.ndarray,
    *,
    lambda_data: float = PENDULUM_LAMBDA_DATA,
    lambda_ode: float = PENDULUM_LAMBDA_ODE,
    lambda_ic: float = PENDULUM_LAMBDA_IC,
    lambda_ic_vel: float = PENDULUM_LAMBDA_IC_VEL,
    n_epochs: int = PENDULUM_NUM_EPOCHS,
    lr: float = PENDULUM_LR,
    n_hidden: int = PENDULUM_N_HIDDEN,
    n_phys: int = PENDULUM_N_PHYS,
    include_initial_frame: bool = True,
    frame_every_epochs: int = PENDULUM_GIF_FRAME_EVERY_EPOCHS,
    gif_dpi: int = PENDULUM_GIF_DPI,
    train_seed: int = PENDULUM_TRAIN_SEED,
) -> list[Image.Image]:
    if frame_every_epochs < 1:
        raise ValueError(f"frame_every_epochs должно быть >= 1, получено {frame_every_epochs}")

    t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
    theta_data_tensor = torch.tensor(theta_data, dtype=torch.float32).view(-1, 1)
    t_phys = torch.linspace(params.t_min, params.t_max, n_phys, dtype=torch.float32).view(-1, 1)
    t_line = _evaluation_grid(params)
    warmup_epochs = n_epochs // 2

    model = _create_pendulum_mlp(
        params,
        n_hidden=n_hidden,
        seed=train_seed,
        use_fourier_features=PENDULUM_PINN_USE_FOURIER,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fig, ax, pred_line = _setup_pendulum_gif_axes(
        t_data,
        theta_data,
        t_analytical,
        theta_analytical,
        t_line,
        model,
        params,
        title=_format_pinn_lambda_title(
            0,
            n_epochs,
            lambda_data,
            lambda_ode,
            lambda_ic,
            lambda_ic_vel,
        ),
        prediction_label="PINN",
        prediction_color=PENDULUM_PINN_FOURIER_COLOR,
    )

    frames: list[Image.Image] = []

    def capture(epoch_label: int) -> None:
        model.eval()
        _update_prediction_line(model, t_line, pred_line)
        ax.set_title(
            _format_pinn_lambda_title(
                epoch_label,
                n_epochs,
                lambda_data,
                lambda_ode,
                lambda_ic,
                lambda_ic_vel,
            )
        )
        frames.append(_figure_to_image(fig, dpi=gif_dpi))
        model.train()

    def should_capture(epoch: int) -> bool:
        return epoch % frame_every_epochs == 0 or epoch == n_epochs - 1

    if include_initial_frame:
        capture(0)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        l_data = _data_loss(model, t_data_tensor, theta_data_tensor)
        l_ode = _physics_loss(model, t_phys, params)
        l_ic = _initial_condition_loss(model, params)
        l_ic_vel = _initial_velocity_loss(model, params)

        if epoch < warmup_epochs:
            loss = (
                lambda_data * l_data
                + lambda_ic * l_ic
                + lambda_ic_vel * l_ic_vel
            )
        else:
            loss = (
                lambda_data * l_data
                + lambda_ode * l_ode
                + lambda_ic * l_ic
                + lambda_ic_vel * l_ic_vel
            )

        loss.backward()
        optimizer.step()
        if should_capture(epoch):
            capture(epoch + 1)

    plt.close(fig)
    return frames


@dataclass(frozen=True)
class _CompareModelSpec:
    label: str
    color: str
    use_fourier_features: bool
    use_pinn_loss: bool


@dataclass
class _CompareTrack:
    spec: _CompareModelSpec
    model: PendulumMLP
    optimizer: torch.optim.Optimizer
    line: object


COMPARE_1_SPECS = (
    _CompareModelSpec("Базовая модель", PENDULUM_VANILLA_COLOR, False, False),
    _CompareModelSpec("PINN", PENDULUM_PINN_PLAIN_COLOR, False, True),
)

COMPARE_2_SPECS = (
    _CompareModelSpec("Базовая модель", PENDULUM_VANILLA_COLOR, False, False),
    _CompareModelSpec("PINN", PENDULUM_PINN_PLAIN_COLOR, False, True),
    _CompareModelSpec("PINN (Фурье)", PENDULUM_PINN_FOURIER_COLOR, True, True),
)


def _pinn_loss(
    model: PendulumMLP,
    t_data_tensor: torch.Tensor,
    theta_data_tensor: torch.Tensor,
    t_phys: torch.Tensor,
    params: PendulumParams,
    epoch: int,
    warmup_epochs: int,
    lambda_data: float,
    lambda_ode: float,
    lambda_ic: float,
    lambda_ic_vel: float,
) -> torch.Tensor:
    l_data = _data_loss(model, t_data_tensor, theta_data_tensor)
    l_ode = _physics_loss(model, t_phys, params)
    l_ic = _initial_condition_loss(model, params)
    l_ic_vel = _initial_velocity_loss(model, params)
    if epoch < warmup_epochs:
        return lambda_data * l_data + lambda_ic * l_ic + lambda_ic_vel * l_ic_vel
    return (
        lambda_data * l_data
        + lambda_ode * l_ode
        + lambda_ic * l_ic
        + lambda_ic_vel * l_ic_vel
    )


def _setup_compare_gif_axes(
    t_data: np.ndarray,
    theta_data: np.ndarray,
    t_analytical: np.ndarray,
    theta_analytical: np.ndarray,
    t_line: np.ndarray,
    tracks: list[_CompareTrack],
    params: PendulumParams,
    *,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    y_min, y_max = _plot_y_limits(theta_analytical)

    fig, ax = plt.subplots(figsize=(PENDULUM_FIG_WIDTH, PENDULUM_FIG_HEIGHT))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.12)

    ax.scatter(
        t_data,
        theta_data,
        color=PENDULUM_DATA_COLOR,
        s=18,
        label="Данные (выборка)",
        zorder=4,
    )
    ax.plot(
        t_analytical,
        theta_analytical,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Аналитическое решение",
        zorder=2,
    )
    for track in tracks:
        (line,) = ax.plot(
            t_line,
            _predict_mlp(track.model, t_line),
            color=track.spec.color,
            linewidth=1.2,
            label=track.spec.label,
            zorder=3,
        )
        track.line = line

    _apply_pendulum_plot_style(ax, params.t_min, params.t_max, y_min, y_max)
    ax.set_title(title)
    return fig, ax


def _train_compare_gif_frames(
    params: PendulumParams,
    t_data: np.ndarray,
    theta_data: np.ndarray,
    t_analytical: np.ndarray,
    theta_analytical: np.ndarray,
    model_specs: tuple[_CompareModelSpec, ...],
    *,
    lambda_data: float = PENDULUM_LAMBDA_DATA,
    lambda_ode: float = PENDULUM_LAMBDA_ODE,
    lambda_ic: float = PENDULUM_LAMBDA_IC,
    lambda_ic_vel: float = PENDULUM_LAMBDA_IC_VEL,
    n_epochs: int = PENDULUM_NUM_EPOCHS,
    lr: float = PENDULUM_LR,
    n_hidden: int = PENDULUM_N_HIDDEN,
    n_phys: int = PENDULUM_N_PHYS,
    include_initial_frame: bool = True,
    frame_every_epochs: int = PENDULUM_GIF_FRAME_EVERY_EPOCHS,
    gif_dpi: int = PENDULUM_GIF_DPI,
    train_seed: int = PENDULUM_TRAIN_SEED,
) -> list[Image.Image]:
    if frame_every_epochs < 1:
        raise ValueError(f"frame_every_epochs должно быть >= 1, получено {frame_every_epochs}")

    t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
    theta_data_tensor = torch.tensor(theta_data, dtype=torch.float32).view(-1, 1)
    t_phys = torch.linspace(params.t_min, params.t_max, n_phys, dtype=torch.float32).view(-1, 1)
    t_line = _evaluation_grid(params)
    warmup_epochs = n_epochs // 2

    tracks: list[_CompareTrack] = []
    for spec in model_specs:
        model = _create_pendulum_mlp(
            params,
            n_hidden=n_hidden,
            seed=train_seed,
            use_fourier_features=spec.use_fourier_features,
        )
        tracks.append(
            _CompareTrack(
                spec=spec,
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=lr),
                line=None,
            )
        )

    fig, ax = _setup_compare_gif_axes(
        t_data,
        theta_data,
        t_analytical,
        theta_analytical,
        t_line,
        tracks,
        params,
        title=f"Эпоха 0/{n_epochs}",
    )

    frames: list[Image.Image] = []

    def capture(epoch_label: int) -> None:
        for track in tracks:
            track.model.eval()
            track.line.set_data(t_line, _predict_mlp(track.model, t_line))
        ax.set_title(f"Эпоха {epoch_label}/{n_epochs}")
        frames.append(_figure_to_image(fig, dpi=gif_dpi))
        for track in tracks:
            track.model.train()

    def should_capture(epoch: int) -> bool:
        return epoch % frame_every_epochs == 0 or epoch == n_epochs - 1

    if include_initial_frame:
        capture(0)

    for track in tracks:
        track.model.train()

    for epoch in range(n_epochs):
        for track in tracks:
            track.optimizer.zero_grad()
            if track.spec.use_pinn_loss:
                loss = _pinn_loss(
                    track.model,
                    t_data_tensor,
                    theta_data_tensor,
                    t_phys,
                    params,
                    epoch,
                    warmup_epochs,
                    lambda_data,
                    lambda_ode,
                    lambda_ic,
                    lambda_ic_vel,
                )
            else:
                loss = _data_loss(track.model, t_data_tensor, theta_data_tensor)
            loss.backward()
            track.optimizer.step()

        if should_capture(epoch):
            capture(epoch + 1)

    plt.close(fig)
    return frames


def _load_training_context(
    csv_path: Path | str,
    t_min: float = PENDULUM_T_MIN,
    t_max: float = PENDULUM_T_MAX,
    n_data: int = PENDULUM_N_DATA,
    data_seed: int = PENDULUM_DATA_SEED,
) -> tuple[PendulumParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = get_pendulum_params(csv_path, t_max=t_max, t_start=t_min)
    t_data, theta_data = sample_pendulum_data(params, n_data, data_seed)
    t_analytical, theta_analytical = _analytical_plot_lines(params)
    return params, t_data, theta_data, t_analytical, theta_analytical


def plot_pendulum_vanilla_gif(
    csv_path: Path | str = DEFAULT_DATA_CSV,
    output_path: Path | str = FIGS_DIR / "pendulum_vanilla.gif",
    t_min: float = PENDULUM_T_MIN,
    t_max: float = PENDULUM_T_MAX,
    n_data: int = PENDULUM_N_DATA,
    data_seed: int = PENDULUM_DATA_SEED,
    n_epochs: int = PENDULUM_NUM_EPOCHS,
    lr: float = PENDULUM_LR,
    n_hidden: int = PENDULUM_N_HIDDEN,
    frame_duration_ms: int | None = None,
    frame_every_epochs: int = PENDULUM_GIF_FRAME_EVERY_EPOCHS,
    include_initial_frame: bool = True,
    train_seed: int = PENDULUM_TRAIN_SEED,
    show: bool = False,
) -> Path:
    """GIF обучения базовой модели (только data loss)."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    params, t_data, theta_data, t_analytical, theta_analytical = _load_training_context(
        csv_path,
        t_min=t_min,
        t_max=t_max,
        n_data=n_data,
        data_seed=data_seed,
    )
    frames = _train_vanilla_gif_frames(
        params,
        t_data,
        theta_data,
        t_analytical,
        theta_analytical,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        include_initial_frame=include_initial_frame,
        frame_every_epochs=frame_every_epochs,
        train_seed=train_seed,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_pendulum_pinn_gif(
    csv_path: Path | str = DEFAULT_DATA_CSV,
    output_path: Path | str = FIGS_DIR / "pendulum_pinn.gif",
    t_min: float = PENDULUM_T_MIN,
    t_max: float = PENDULUM_T_MAX,
    n_data: int = PENDULUM_N_DATA,
    data_seed: int = PENDULUM_DATA_SEED,
    lambda_data: float = PENDULUM_LAMBDA_DATA,
    lambda_ode: float = PENDULUM_LAMBDA_ODE,
    lambda_ic: float = PENDULUM_LAMBDA_IC,
    lambda_ic_vel: float = PENDULUM_LAMBDA_IC_VEL,
    n_epochs: int = PENDULUM_NUM_EPOCHS,
    lr: float = PENDULUM_LR,
    n_hidden: int = PENDULUM_N_HIDDEN,
    n_phys: int = PENDULUM_N_PHYS,
    frame_duration_ms: int | None = None,
    frame_every_epochs: int = PENDULUM_GIF_FRAME_EVERY_EPOCHS,
    include_initial_frame: bool = True,
    train_seed: int = PENDULUM_TRAIN_SEED,
    show: bool = False,
) -> Path:
    """GIF обучения PINN с фиксированными λ из pinn_pendulum.py."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    params, t_data, theta_data, t_analytical, theta_analytical = _load_training_context(
        csv_path,
        t_min=t_min,
        t_max=t_max,
        n_data=n_data,
        data_seed=data_seed,
    )
    frames = _train_pinn_gif_frames(
        params,
        t_data,
        theta_data,
        t_analytical,
        theta_analytical,
        lambda_data=lambda_data,
        lambda_ode=lambda_ode,
        lambda_ic=lambda_ic,
        lambda_ic_vel=lambda_ic_vel,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        n_phys=n_phys,
        include_initial_frame=include_initial_frame,
        frame_every_epochs=frame_every_epochs,
        train_seed=train_seed,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def _plot_compare_gif(
    model_specs: tuple[_CompareModelSpec, ...],
    output_path: Path | str,
    csv_path: Path | str = DEFAULT_DATA_CSV,
    t_min: float = PENDULUM_T_MIN,
    t_max: float = PENDULUM_T_MAX,
    n_data: int = PENDULUM_N_DATA,
    data_seed: int = PENDULUM_DATA_SEED,
    lambda_data: float = PENDULUM_LAMBDA_DATA,
    lambda_ode: float = PENDULUM_LAMBDA_ODE,
    lambda_ic: float = PENDULUM_LAMBDA_IC,
    lambda_ic_vel: float = PENDULUM_LAMBDA_IC_VEL,
    n_epochs: int = PENDULUM_NUM_EPOCHS,
    lr: float = PENDULUM_LR,
    n_hidden: int = PENDULUM_N_HIDDEN,
    n_phys: int = PENDULUM_N_PHYS,
    frame_duration_ms: int | None = None,
    frame_every_epochs: int = PENDULUM_GIF_FRAME_EVERY_EPOCHS,
    include_initial_frame: bool = True,
    train_seed: int = PENDULUM_TRAIN_SEED,
    show: bool = False,
) -> Path:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    params, t_data, theta_data, t_analytical, theta_analytical = _load_training_context(
        csv_path,
        t_min=t_min,
        t_max=t_max,
        n_data=n_data,
        data_seed=data_seed,
    )
    frames = _train_compare_gif_frames(
        params,
        t_data,
        theta_data,
        t_analytical,
        theta_analytical,
        model_specs,
        lambda_data=lambda_data,
        lambda_ode=lambda_ode,
        lambda_ic=lambda_ic,
        lambda_ic_vel=lambda_ic_vel,
        n_epochs=n_epochs,
        lr=lr,
        n_hidden=n_hidden,
        n_phys=n_phys,
        include_initial_frame=include_initial_frame,
        frame_every_epochs=frame_every_epochs,
        train_seed=train_seed,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_pendulum_compare_1_gif(
    csv_path: Path | str = DEFAULT_DATA_CSV,
    output_path: Path | str = FIGS_DIR / "pendulum_compare_1.gif",
    **kwargs,
) -> Path:
    """GIF: базовая модель + PINN без Fourier."""
    return _plot_compare_gif(COMPARE_1_SPECS, output_path, csv_path=csv_path, **kwargs)


def plot_pendulum_compare_2_gif(
    csv_path: Path | str = DEFAULT_DATA_CSV,
    output_path: Path | str = FIGS_DIR / "pendulum_compare_2.gif",
    **kwargs,
) -> Path:
    """GIF: базовая модель + PINN + PINN (Фурье)."""
    return _plot_compare_gif(COMPARE_2_SPECS, output_path, csv_path=csv_path, **kwargs)


PLOTS: dict[str, Callable[..., Path]] = {
    "pendulum_vanilla_gif": plot_pendulum_vanilla_gif,
    "pendulum_pinn_gif": plot_pendulum_pinn_gif,
    "pendulum_compare_1_gif": plot_pendulum_compare_1_gif,
    "pendulum_compare_2_gif": plot_pendulum_compare_2_gif,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GIF обучения маятника (vanilla / PINN)")
    parser.add_argument(
        "plot",
        nargs="?",
        default="pendulum_vanilla_gif",
        choices=sorted(PLOTS),
        help="Имя анимации для генерации",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--t-min", type=float, default=PENDULUM_T_MIN)
    parser.add_argument("--t-max", type=float, default=PENDULUM_T_MAX)
    parser.add_argument("--n-data", type=int, default=PENDULUM_N_DATA)
    parser.add_argument("--data-seed", type=int, default=PENDULUM_DATA_SEED)
    parser.add_argument("--train-seed", type=int, default=PENDULUM_TRAIN_SEED)
    parser.add_argument("--train-epochs", type=int, default=PENDULUM_NUM_EPOCHS)
    parser.add_argument("--frame-every-epochs", type=int, default=PENDULUM_GIF_FRAME_EVERY_EPOCHS)
    parser.add_argument("--frame-duration-ms", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    plot_fn = PLOTS[args.plot]
    kwargs: dict = {
        "csv_path": args.csv,
        "t_min": args.t_min,
        "t_max": args.t_max,
        "n_data": args.n_data,
        "data_seed": args.data_seed,
        "train_seed": args.train_seed,
        "n_epochs": args.train_epochs,
        "frame_every_epochs": args.frame_every_epochs,
        "show": args.show,
    }
    if args.output is not None:
        kwargs["output_path"] = args.output
    if args.frame_duration_ms is not None:
        kwargs["frame_duration_ms"] = args.frame_duration_ms

    result = plot_fn(**kwargs)
    print(f"Сохранено: {result}")

    n_frames = args.train_epochs // args.frame_every_epochs + 1
    if args.train_epochs % args.frame_every_epochs != 0:
        n_frames += 1
    frame_duration_ms = args.frame_duration_ms
    if frame_duration_ms is None:
        frame_duration_ms = _gif_frame_duration_ms(n_frames)
    total_duration_s = (
        (n_frames - 1) * frame_duration_ms + PENDULUM_GIF_FINAL_PAUSE_MS
    ) / 1000
    print(
        f"Кадров: ~{n_frames}, обучение: {args.train_epochs} эпох, "
        f"~{total_duration_s:.1f} с"
    )
