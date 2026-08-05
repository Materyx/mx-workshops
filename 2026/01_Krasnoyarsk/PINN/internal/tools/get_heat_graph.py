from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[2]
PINN_DIR = ROOT
ASSETS = ROOT / "assets"
FIGS_DIR = ASSETS / "heat_transfer_2d"

# Геометрия и физика — как в pinn_heat_transfer_2d.py
PLATE_W = 0.25
PLATE_H = 0.1
K_THERMAL = 237.0
RHO = 2700.0
C_HEAT = 900.0
ALPHA = K_THERMAL / (RHO * C_HEAT)

T_INIT = 20.0
T_LEFT = 20.0
T_RIGHT = 70.0

T_MIN = 0.0
T_MAX = 400.0
T_PLOT_VALUES = [1, 40, 100, 400, 800]

SENSOR_SEED_BASE = 4
N_SENSORS = 15
NOISE_LEVEL = 2.0

HEAT_N_HIDDEN = 32
HEAT_LR = 1e-3
HEAT_NUM_EPOCHS = 6000
HEAT_LAMBDA_DATA = 2.0
HEAT_LAMBDA_PDE = 2.0
HEAT_LAMBDA_IC = 2.0
HEAT_LAMBDA_BC = 2.0

HEAT_PLOT_NX = 200
HEAT_PLOT_NY = 80
HEAT_T_VMIN = 20.0
HEAT_T_VMAX = 70.0
HEAT_ERR_CMAP = "terrain"
HEAT_ERR_VMIN = 0.0

HEAT_GIF_DURATION_MS = 10_000
HEAT_GIF_FINAL_PAUSE_MS = 5_000
HEAT_FEM_N_FRAMES = 80
HEAT_TRAIN_SEED = 42
HEAT_GIF_DPI = 100
HEAT_FIG_WIDTH = 6.5
HEAT_FIG_HEIGHT = 2.8
HEAT_MEAN_ERR_DT = 1.0
HEAT_MEAN_ERR_FIG_HEIGHT = HEAT_FIG_HEIGHT * 0.72


class HeatPINN(nn.Module):
    """MLP T(x, y, t) — архитектура из pinn_heat_transfer_2d.py."""

    def __init__(self, n_hidden: int = HEAT_N_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_norm = x / PLATE_W
        y_norm = y / PLATE_H
        t_norm = t / T_MAX
        inputs = torch.cat([x_norm, y_norm, t_norm], dim=1)
        return self.net(inputs)


def _set_train_seed(seed: int = HEAT_TRAIN_SEED) -> None:
    torch.manual_seed(seed)


def _create_heat_pinn(n_hidden: int = HEAT_N_HIDDEN, seed: int = HEAT_TRAIN_SEED) -> HeatPINN:
    _set_train_seed(seed)
    return HeatPINN(n_hidden=n_hidden)


def _figure_to_image(fig: plt.Figure, dpi: int = HEAT_GIF_DPI) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    buffer.seek(0)
    return Image.open(buffer).copy()


def _gif_frame_duration_ms(n_frames: int, total_duration_ms: int = HEAT_GIF_DURATION_MS) -> int:
    return max(1, round(total_duration_ms / n_frames))


def _save_gif(
    frames: list[Image.Image],
    output_path: Path,
    frame_duration_ms: int | None = None,
    *,
    total_duration_ms: int = HEAT_GIF_DURATION_MS,
    final_pause_ms: int = HEAT_GIF_FINAL_PAUSE_MS,
) -> None:
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


def build_fem_mesh(nx_cells: int, ny_cells: int):
    nx_nodes = nx_cells + 1
    ny_nodes = ny_cells + 1
    x_nodes = np.linspace(0.0, PLATE_W, nx_nodes)
    y_nodes = np.linspace(0.0, PLATE_H, ny_nodes)

    def node_id(i: int, j: int) -> int:
        return i + j * nx_nodes

    nodes = np.zeros((nx_nodes * ny_nodes, 2))
    for j in range(ny_nodes):
        for i in range(nx_nodes):
            nid = node_id(i, j)
            nodes[nid, 0] = x_nodes[i]
            nodes[nid, 1] = y_nodes[j]

    elements = []
    for j in range(ny_cells):
        for i in range(nx_cells):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i + 1, j + 1)
            n3 = node_id(i, j + 1)
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

    left_nodes = [node_id(0, j) for j in range(ny_nodes)]
    right_nodes = [node_id(nx_cells, j) for j in range(ny_nodes)]
    return nodes, np.array(elements), x_nodes, y_nodes, left_nodes, right_nodes


def triangle_matrices(coords: np.ndarray):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    b_mat = np.vstack([b, c]) / (2.0 * area)

    k_local = ALPHA * area * (b_mat.T @ b_mat)
    m_local = (area / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    return k_local, m_local


def assemble_fem_system(nodes: np.ndarray, elements: np.ndarray):
    n_nodes = nodes.shape[0]
    k_mat = sparse.lil_matrix((n_nodes, n_nodes))
    m_mat = sparse.lil_matrix((n_nodes, n_nodes))

    for elem in elements:
        coords = nodes[elem]
        k_loc, m_loc = triangle_matrices(coords)
        for a in range(3):
            for b in range(3):
                k_mat[elem[a], elem[b]] += k_loc[a, b]
                m_mat[elem[a], elem[b]] += m_loc[a, b]

    return k_mat.tocsr(), m_mat.tocsr()


def apply_dirichlet(a_mat: sparse.spmatrix, b_vec: np.ndarray, dirichlet_map: dict[int, float]):
    a_lil = a_mat.tolil()
    for node, value in dirichlet_map.items():
        a_lil.rows[node] = [node]
        a_lil.data[node] = [1.0]
        b_vec[node] = value
    return a_lil.tocsr(), b_vec


def run_fem_solver():
    nx_cells, ny_cells = 50, 20
    dt = 2.0
    nodes, elements, x_nodes, y_nodes, left_nodes, right_nodes = build_fem_mesh(
        nx_cells, ny_cells
    )
    k_mat, m_mat = assemble_fem_system(nodes, elements)

    dirichlet = {node: T_LEFT for node in left_nodes}
    dirichlet.update({node: T_RIGHT for node in right_nodes})

    n_nodes = nodes.shape[0]
    temperature = np.full(n_nodes, T_INIT)

    snapshots = {0.0: temperature.copy()}
    t_current = 0.0
    next_snapshot_idx = 0
    snapshot_targets = sorted(t for t in T_PLOT_VALUES if t > 0)

    a_base = (m_mat / dt + k_mat).tocsc()
    m_dt = (m_mat / dt).tocsc()

    while t_current < T_MAX - 1e-9:
        t_current += dt
        b_vec = m_dt @ temperature
        a_mat, b_vec = apply_dirichlet(a_base.copy(), b_vec.copy(), dirichlet)
        temperature = spsolve(a_mat, b_vec)

        if (
            next_snapshot_idx < len(snapshot_targets)
            and t_current >= snapshot_targets[next_snapshot_idx] - 1e-9
        ):
            t_snap = snapshot_targets[next_snapshot_idx]
            snapshots[t_snap] = temperature.copy()
            next_snapshot_idx += 1

    ny_nodes = ny_cells + 1
    nx_nodes = nx_cells + 1
    interpolators = {}
    for t_snap, t_nodes in snapshots.items():
        t_grid = t_nodes.reshape(ny_nodes, nx_nodes)
        interpolators[t_snap] = RegularGridInterpolator(
            (y_nodes, x_nodes),
            t_grid,
            bounds_error=False,
            fill_value=None,
        )

    return np.array(sorted(snapshots.keys())), interpolators


class FemReference:
    def __init__(self):
        self.times, self.interpolators = run_fem_solver()

    def solution(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)

        shape = np.broadcast_shapes(x.shape, y.shape, t.shape)
        x_b = np.broadcast_to(x, shape).ravel()
        y_b = np.broadcast_to(y, shape).ravel()
        t_b = np.broadcast_to(t, shape).ravel()

        pts = np.column_stack([y_b, x_b])
        t_snap = np.array([self.interpolators[t_s](pts) for t_s in self.times])

        result = np.zeros(len(x_b))
        for i in range(len(x_b)):
            result[i] = np.interp(t_b[i], self.times, t_snap[:, i])

        return result.reshape(shape)


def _build_plot_grid() -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    x_grid = np.linspace(0, PLATE_W, HEAT_PLOT_NX)
    y_grid = np.linspace(0, PLATE_H, HEAT_PLOT_NY)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    x_flat = torch.tensor(x_mesh.flatten(), dtype=torch.float32).view(-1, 1)
    y_flat = torch.tensor(y_mesh.flatten(), dtype=torch.float32).view(-1, 1)
    return x_mesh, y_mesh, x_flat, y_flat


def _build_sensor_data_by_time(
    fem: FemReference,
) -> dict[float, dict[str, np.ndarray]]:
    """Пробы по моментам времени: x, y, T (зашумлённые)."""
    sensor_by_time: dict[float, dict[str, np.ndarray]] = {}
    for t_val in T_PLOT_VALUES:
        rng = np.random.default_rng(SENSOR_SEED_BASE + int(t_val))
        sx = rng.uniform(0.0, PLATE_W, N_SENSORS)
        sy = rng.uniform(0.0, PLATE_H, N_SENSORS)
        t_exact = fem.solution(sx, sy, t_val)
        t_noisy = t_exact + NOISE_LEVEL * rng.standard_normal(N_SENSORS)
        sensor_by_time[float(t_val)] = {"x": sx, "y": sy, "T": t_noisy}
    return sensor_by_time


def _prepare_sensor_data(fem: FemReference):
    sensor_by_time = _build_sensor_data_by_time(fem)
    x_list: list[float] = []
    y_list: list[float] = []
    t_list: list[float] = []
    t_list_values: list[float] = []

    for t_val in T_PLOT_VALUES:
        sd = sensor_by_time[float(t_val)]
        x_list.extend(sd["x"])
        y_list.extend(sd["y"])
        t_list.extend([t_val] * N_SENSORS)
        t_list_values.extend(sd["T"])

    x_data = np.array(x_list, dtype=np.float32)
    y_data = np.array(y_list, dtype=np.float32)
    t_data = np.array(t_list, dtype=np.float32)
    t_data_values = np.array(t_list_values, dtype=np.float32)

    return (
        torch.tensor(x_data, dtype=torch.float32).view(-1, 1),
        torch.tensor(y_data, dtype=torch.float32).view(-1, 1),
        torch.tensor(t_data, dtype=torch.float32).view(-1, 1),
        torch.tensor(t_data_values, dtype=torch.float32).view(-1, 1),
    )


def _derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def _sample_interior_points(n_points: int):
    x = torch.rand(n_points, 1) * PLATE_W
    y = torch.rand(n_points, 1) * PLATE_H
    t = torch.rand(n_points, 1) * T_MAX
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    return x, y, t


def _sample_initial_points(n_points: int):
    x = torch.rand(n_points, 1) * PLATE_W
    y = torch.rand(n_points, 1) * PLATE_H
    t = torch.zeros(n_points, 1)
    return x, y, t


def _sample_boundary_points(n_points: int):
    n_each = n_points // 4
    t_bc_min = 1.0

    # x = 0: T = T_LEFT
    x_left = torch.zeros(n_each, 1)
    y_left = torch.rand(n_each, 1) * PLATE_H
    t_left = t_bc_min + torch.rand(n_each, 1) * (T_MAX - t_bc_min)

    # x = PLATE_W: T = T_RIGHT
    x_right = torch.full((n_each, 1), PLATE_W)
    y_right = torch.rand(n_each, 1) * PLATE_H
    t_right = t_bc_min + torch.rand(n_each, 1) * (T_MAX - t_bc_min)

    # y = 0: dT/dy = 0
    x_bottom = torch.rand(n_each, 1) * PLATE_W
    y_bottom = torch.zeros(n_each, 1)
    t_bottom = torch.rand(n_each, 1) * T_MAX
    y_bottom.requires_grad_(True)

    # y = PLATE_H: dT/dy = 0
    x_top = torch.rand(n_each, 1) * PLATE_W
    y_top = torch.full((n_each, 1), PLATE_H)
    t_top = torch.rand(n_each, 1) * T_MAX
    y_top.requires_grad_(True)

    return (
        x_left,
        y_left,
        t_left,
        x_right,
        y_right,
        t_right,
        x_bottom,
        y_bottom,
        t_bottom,
        x_top,
        y_top,
        t_top,
    )


def _physics_loss(model: HeatPINN, n_points: int = 2000) -> torch.Tensor:
    x, y, t = _sample_interior_points(n_points)
    t_pred = model(x, y, t)
    dt_pred = _derivative(t_pred, t)
    dx_pred = _derivative(t_pred, x)
    d2x_pred = _derivative(dx_pred, x)
    dy_pred = _derivative(t_pred, y)
    d2y_pred = _derivative(dy_pred, y)
    residual = dt_pred - ALPHA * (d2x_pred + d2y_pred)
    return torch.mean(residual**2)


def _initial_condition_loss(model: HeatPINN, n_points: int = 500) -> torch.Tensor:
    x, y, t = _sample_initial_points(n_points)
    t_pred = model(x, y, t)
    return torch.mean((t_pred - T_INIT) ** 2)


def _boundary_condition_loss(model: HeatPINN, n_points: int = 400) -> torch.Tensor:
    (
        x_left,
        y_left,
        t_left,
        x_right,
        y_right,
        t_right,
        x_bottom,
        y_bottom,
        t_bottom,
        x_top,
        y_top,
        t_top,
    ) = _sample_boundary_points(n_points)

    t_left_pred = model(x_left, y_left, t_left)
    loss_left = torch.mean((t_left_pred - T_LEFT) ** 2)

    t_right_pred = model(x_right, y_right, t_right)
    loss_right = torch.mean((t_right_pred - T_RIGHT) ** 2)

    t_bottom_pred = model(x_bottom, y_bottom, t_bottom)
    dtdy_bottom = _derivative(t_bottom_pred, y_bottom)
    loss_bottom = torch.mean(dtdy_bottom**2)

    t_top_pred = model(x_top, y_top, t_top)
    dtdy_top = _derivative(t_top_pred, y_top)
    loss_top = torch.mean(dtdy_top**2)

    return loss_left + loss_right + loss_bottom + loss_top


def _data_loss(
    model: HeatPINN,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    t_data: torch.Tensor,
    t_values: torch.Tensor,
) -> torch.Tensor:
    t_pred = model(x_data, y_data, t_data)
    return torch.mean((t_pred - t_values) ** 2)


def _predict_field(
    model: HeatPINN,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    x_flat: torch.Tensor,
    y_flat: torch.Tensor,
    t_val: float,
) -> np.ndarray:
    t_flat = torch.full_like(x_flat, t_val)
    with torch.no_grad():
        t_field = model(x_flat, y_flat, t_flat).numpy().reshape(x_mesh.shape)
    return t_field


def _draw_single_field_frame(
    ax: plt.Axes,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    field: np.ndarray,
    title: str,
) -> None:
    ax.clear()
    ax.pcolormesh(
        x_mesh,
        y_mesh,
        field,
        cmap="jet",
        vmin=HEAT_T_VMIN,
        vmax=HEAT_T_VMAX,
        shading="auto",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, PLATE_W)
    ax.set_ylim(0, PLATE_H)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(title)


def _draw_error_field_frame(
    ax: plt.Axes,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    err_field: np.ndarray,
    title: str,
    err_vmax: float,
) -> None:
    ax.clear()
    ax.pcolormesh(
        x_mesh,
        y_mesh,
        err_field,
        cmap=HEAT_ERR_CMAP,
        vmin=HEAT_ERR_VMIN,
        vmax=err_vmax,
        shading="auto",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, PLATE_W)
    ax.set_ylim(0, PLATE_H)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(title)


def _active_sensor_batch(
    t_frame: float,
    sensor_by_time: dict[float, dict[str, np.ndarray]],
) -> tuple[float, dict[str, np.ndarray]] | None:
    """Набор проб, актуальный для текущего кадра (только последний замер)."""
    active_t: float | None = None
    for t_val in sorted(sensor_by_time):
        if t_frame >= t_val - 1e-9:
            active_t = t_val
    if active_t is None:
        return None
    return active_t, sensor_by_time[active_t]


def _draw_probe_frame(
    ax: plt.Axes,
    batch: dict[str, np.ndarray] | None,
    title: str,
) -> None:
    ax.clear()
    ax.plot(
        [0, PLATE_W, PLATE_W, 0, 0],
        [0, 0, PLATE_H, PLATE_H, 0],
        "k-",
        lw=1,
    )
    if batch is not None:
        ax.scatter(
            batch["x"],
            batch["y"],
            c=batch["T"],
            cmap="jet",
            vmin=HEAT_T_VMIN,
            vmax=HEAT_T_VMAX,
            s=60,
            edgecolors="k",
            linewidths=0.5,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, PLATE_W)
    ax.set_ylim(0, PLATE_H)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(title)


def _train_vanilla_model(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    t_data: torch.Tensor,
    t_values: torch.Tensor,
    n_epochs: int = HEAT_NUM_EPOCHS,
) -> HeatPINN:
    model = _create_heat_pinn()
    optimizer = torch.optim.Adam(model.parameters(), lr=HEAT_LR)
    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = _data_loss(model, x_data, y_data, t_data, t_values)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _train_pinn_model(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    t_data: torch.Tensor,
    t_values: torch.Tensor,
    n_epochs: int = HEAT_NUM_EPOCHS,
) -> HeatPINN:
    model = _create_heat_pinn()
    optimizer = torch.optim.Adam(model.parameters(), lr=HEAT_LR)
    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = (
            HEAT_LAMBDA_DATA
            * _data_loss(model, x_data, y_data, t_data, t_values)
            + HEAT_LAMBDA_PDE * _physics_loss(model)
            + HEAT_LAMBDA_IC * _initial_condition_loss(model)
            + HEAT_LAMBDA_BC * _boundary_condition_loss(model)
        )
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _model_gif_frames(
    model: HeatPINN,
    title_prefix: str,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    x_flat: torch.Tensor,
    y_flat: torch.Tensor,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    gif_dpi: int = HEAT_GIF_DPI,
) -> list[Image.Image]:
    times = np.linspace(0.0, t_end, n_frames)
    fig, ax = plt.subplots(figsize=(HEAT_FIG_WIDTH, HEAT_FIG_HEIGHT))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.14)

    frames: list[Image.Image] = []
    for t_val in times:
        field = _predict_field(model, x_mesh, y_mesh, x_flat, y_flat, float(t_val))
        _draw_single_field_frame(
            ax,
            x_mesh,
            y_mesh,
            field,
            f"{title_prefix} | t = {t_val:.1f} с",
        )
        frames.append(_figure_to_image(fig, dpi=gif_dpi))

    plt.close(fig)
    return frames


def _fem_gif_frames(
    fem: FemReference,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    gif_dpi: int = HEAT_GIF_DPI,
) -> list[Image.Image]:
    times = np.linspace(0.0, t_end, n_frames)
    fig, ax = plt.subplots(figsize=(HEAT_FIG_WIDTH, HEAT_FIG_HEIGHT))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.14)

    frames: list[Image.Image] = []
    for t_val in times:
        field = fem.solution(x_mesh, y_mesh, t_val)
        _draw_single_field_frame(ax, x_mesh, y_mesh, field, f"МКЭ | t = {t_val:.1f} с")
        frames.append(_figure_to_image(fig, dpi=gif_dpi))

    plt.close(fig)
    return frames


def _probe_gif_frames(
    sensor_by_time: dict[float, dict[str, np.ndarray]],
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    gif_dpi: int = HEAT_GIF_DPI,
) -> list[Image.Image]:
    times = np.linspace(0.0, t_end, n_frames)
    fig, ax = plt.subplots(figsize=(HEAT_FIG_WIDTH, HEAT_FIG_HEIGHT))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.14)

    frames: list[Image.Image] = []
    for t_val in times:
        active = _active_sensor_batch(float(t_val), sensor_by_time)
        if active is None:
            batch = None
            title = f"Пробы | t = {t_val:.1f} с"
        else:
            active_t, batch = active
            title = f"Пробы | t = {t_val:.1f} с (замер при t = {active_t:.0f} с)"
        _draw_probe_frame(ax, batch, title)
        frames.append(_figure_to_image(fig, dpi=gif_dpi))

    plt.close(fig)
    return frames


def _error_gif_frames(
    fem: FemReference,
    field_fn: Callable[[float], np.ndarray],
    title_prefix: str,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    err_vmax: float,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    gif_dpi: int = HEAT_GIF_DPI,
) -> list[Image.Image]:
    times = np.linspace(0.0, t_end, n_frames)
    fig, ax = plt.subplots(figsize=(HEAT_FIG_WIDTH, HEAT_FIG_HEIGHT))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.14)

    frames: list[Image.Image] = []
    for t_val in times:
        t_true = fem.solution(x_mesh, y_mesh, float(t_val))
        field = field_fn(float(t_val))
        err_field = np.abs(field - t_true)
        _draw_error_field_frame(
            ax,
            x_mesh,
            y_mesh,
            err_field,
            f"{title_prefix} | t = {t_val:.1f} с",
            err_vmax,
        )
        frames.append(_figure_to_image(fig, dpi=gif_dpi))

    plt.close(fig)
    return frames


_FEM_CACHE: FemReference | None = None
_VANILLA_MODEL_CACHE: dict[int, HeatPINN] = {}
_PINN_MODEL_CACHE: dict[int, HeatPINN] = {}
_ERR_VMAX_CACHE: dict[int, float] = {}
_MEAN_ERR_CURVE_CACHE: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
_MEAN_ERR_YMAX_CACHE: dict[int, float] = {}


def _get_vanilla_model(fem: FemReference, n_epochs: int = HEAT_NUM_EPOCHS) -> HeatPINN:
    if n_epochs not in _VANILLA_MODEL_CACHE:
        sensor_tensors = _prepare_sensor_data(fem)
        _VANILLA_MODEL_CACHE[n_epochs] = _train_vanilla_model(*sensor_tensors, n_epochs=n_epochs)
    return _VANILLA_MODEL_CACHE[n_epochs]


def _get_pinn_model(fem: FemReference, n_epochs: int = HEAT_NUM_EPOCHS) -> HeatPINN:
    if n_epochs not in _PINN_MODEL_CACHE:
        sensor_tensors = _prepare_sensor_data(fem)
        _PINN_MODEL_CACHE[n_epochs] = _train_pinn_model(*sensor_tensors, n_epochs=n_epochs)
    return _PINN_MODEL_CACHE[n_epochs]


def _compute_err_vmax(
    fem: FemReference,
    vanilla_model: HeatPINN,
    pinn_model: HeatPINN,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    x_flat: torch.Tensor,
    y_flat: torch.Tensor,
    t_end: float = max(T_PLOT_VALUES),
    n_samples: int = HEAT_FEM_N_FRAMES,
) -> float:
    times = np.linspace(0.0, t_end, n_samples)
    err_vmax = 0.0
    for t_val in times:
        t_true = fem.solution(x_mesh, y_mesh, float(t_val))
        t_vanilla = _predict_field(
            vanilla_model, x_mesh, y_mesh, x_flat, y_flat, float(t_val)
        )
        t_pinn = _predict_field(pinn_model, x_mesh, y_mesh, x_flat, y_flat, float(t_val))
        err_vmax = max(
            err_vmax,
            float(np.abs(t_vanilla - t_true).max()),
            float(np.abs(t_pinn - t_true).max()),
        )
    return err_vmax


def _get_err_vmax(fem: FemReference, n_epochs: int = HEAT_NUM_EPOCHS) -> float:
    if n_epochs not in _ERR_VMAX_CACHE:
        x_mesh, y_mesh, x_flat, y_flat = _build_plot_grid()
        vanilla_model = _get_vanilla_model(fem, n_epochs)
        pinn_model = _get_pinn_model(fem, n_epochs)
        _ERR_VMAX_CACHE[n_epochs] = _compute_err_vmax(
            fem, vanilla_model, pinn_model, x_mesh, y_mesh, x_flat, y_flat
        )
    return _ERR_VMAX_CACHE[n_epochs]


def _compute_mean_error_curve(
    fem: FemReference,
    field_fn: Callable[[float], np.ndarray],
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    t_end: float = max(T_PLOT_VALUES),
    dt: float = HEAT_MEAN_ERR_DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Средняя |ошибка| по пластине для каждого момента времени."""
    times = np.arange(0.0, t_end + 0.5 * dt, dt)
    mean_errs = np.empty(times.shape[0], dtype=np.float64)
    for i, t_val in enumerate(times):
        t_true = fem.solution(x_mesh, y_mesh, float(t_val))
        field = field_fn(float(t_val))
        mean_errs[i] = np.mean(np.abs(field - t_true))
    return times, mean_errs


def _mean_error_integral(times: np.ndarray, mean_errs: np.ndarray) -> float:
    """Интеграл средней ошибки по времени: площадь под кривой, °C·с."""
    return float(np.trapezoid(mean_errs, times))


def _summarize_mean_error_integrals(
    fem: FemReference,
    n_epochs: int = HEAT_NUM_EPOCHS,
) -> dict[str, float]:
    curves, _ = _get_mean_error_curves(fem, n_epochs=n_epochs)
    labels = {
        "fem": "МКЭ",
        "vanilla": "Базовая модель",
        "pinn": "PINN",
    }
    return {
        labels[key]: _mean_error_integral(times, mean_errs)
        for key, (times, mean_errs) in curves.items()
    }


def _print_mean_error_integrals(n_epochs: int = HEAT_NUM_EPOCHS) -> dict[str, float]:
    fem = _get_fem_reference()
    integrals = _summarize_mean_error_integrals(fem, n_epochs=n_epochs)
    print("Интегральная ошибка (площадь под кривой средней |ошибки|, °C·с):")
    for name, value in integrals.items():
        print(f"  {name}: {value:.2f}")
    return integrals


def _get_mean_error_curves(
    fem: FemReference,
    n_epochs: int = HEAT_NUM_EPOCHS,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    if n_epochs not in _MEAN_ERR_CURVE_CACHE:
        x_mesh, y_mesh, x_flat, y_flat = _build_plot_grid()
        t_end = float(max(T_PLOT_VALUES))

        times_fem, errs_fem = _compute_mean_error_curve(
            fem,
            lambda t_val: fem.solution(x_mesh, y_mesh, t_val),
            x_mesh,
            y_mesh,
            t_end=t_end,
        )
        vanilla_model = _get_vanilla_model(fem, n_epochs)
        times_vanilla, errs_vanilla = _compute_mean_error_curve(
            fem,
            lambda t_val: _predict_field(
                vanilla_model, x_mesh, y_mesh, x_flat, y_flat, t_val
            ),
            x_mesh,
            y_mesh,
            t_end=t_end,
        )
        pinn_model = _get_pinn_model(fem, n_epochs)
        times_pinn, errs_pinn = _compute_mean_error_curve(
            fem,
            lambda t_val: _predict_field(
                pinn_model, x_mesh, y_mesh, x_flat, y_flat, t_val
            ),
            x_mesh,
            y_mesh,
            t_end=t_end,
        )

        _MEAN_ERR_CURVE_CACHE[n_epochs] = {
            "fem": (times_fem, errs_fem),
            "vanilla": (times_vanilla, errs_vanilla),
            "pinn": (times_pinn, errs_pinn),
        }
        y_max = max(float(errs_vanilla.max()), float(errs_pinn.max()))
        _MEAN_ERR_YMAX_CACHE[n_epochs] = y_max if y_max > 1e-9 else 1.0

    return _MEAN_ERR_CURVE_CACHE[n_epochs], _MEAN_ERR_YMAX_CACHE[n_epochs]


def _draw_mean_error_frame(
    ax: plt.Axes,
    times: np.ndarray,
    mean_errs: np.ndarray,
    t_current: float,
    title: str,
    y_max: float,
    t_end: float = max(T_PLOT_VALUES),
) -> None:
    ax.clear()
    mask = times <= t_current + 1e-9
    t_vis = times[mask]
    e_vis = mean_errs[mask]
    if len(t_vis) > 0:
        ax.fill_between(t_vis, e_vis, 0.0, alpha=0.35, color="steelblue")
        ax.plot(t_vis, e_vis, color="steelblue", lw=1.5)
    ax.set_xlim(0.0, t_end)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("t, с")
    ax.set_ylabel("Средняя |ошибка|, °C")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, lw=0.5)


def _mean_error_gif_frames(
    times: np.ndarray,
    mean_errs: np.ndarray,
    title_prefix: str,
    y_max: float,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    gif_dpi: int = HEAT_GIF_DPI,
) -> list[Image.Image]:
    frame_times = np.linspace(0.0, t_end, n_frames)
    fig, ax = plt.subplots(figsize=(HEAT_FIG_WIDTH, HEAT_MEAN_ERR_FIG_HEIGHT))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.82, bottom=0.20)

    frames: list[Image.Image] = []
    for t_frame in frame_times:
        _draw_mean_error_frame(
            ax,
            times,
            mean_errs,
            float(t_frame),
            f"{title_prefix} | t = {t_frame:.1f} с",
            y_max,
            t_end=t_end,
        )
        frames.append(_figure_to_image(fig, dpi=gif_dpi))

    plt.close(fig)
    return frames


def _format_colorbar_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def _error_colorbar_ticks(vmax: float) -> list[float]:
    if vmax <= 1e-9:
        return [0.0]
    step = 10.0 if vmax <= 60 else (20.0 if vmax <= 120 else 50.0)
    ticks = [0.0]
    value = step
    while value < vmax - step * 0.25:
        ticks.append(value)
        value += step
    vmax_rounded = round(vmax, 1)
    if abs(ticks[-1] - vmax_rounded) > 1e-6:
        ticks.append(vmax_rounded)
    return ticks


def _save_horizontal_colorbar_png(
    output_path: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    label: str,
    dpi: int = HEAT_GIF_DPI,
    ticks: list[float] | None = None,
) -> None:
    fig = plt.figure(figsize=(HEAT_FIG_WIDTH, HEAT_FIG_HEIGHT / 4))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    side_margin = 0.12
    bar_width = 1.0 - 2 * side_margin
    cax = fig.add_axes([side_margin, 0.58, bar_width, 0.24], frameon=False)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    if ticks is not None:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([_format_colorbar_tick(t) for t in ticks])
    cbar.ax.tick_params(labelsize=9, pad=1)
    fig.text(0.5, 0.10, label, ha="center", va="center", fontsize=10)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _get_fem_reference() -> FemReference:
    global _FEM_CACHE
    if _FEM_CACHE is None:
        _FEM_CACHE = FemReference()
    return _FEM_CACHE


def plot_heat_fem_gif(
    output_path: Path | str = FIGS_DIR / "heat_fem.gif",
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    frame_duration_ms: int | None = None,
    show: bool = False,
) -> Path:
    """GIF эволюции поля температуры по МКЭ-эталону."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    x_mesh, y_mesh, _, _ = _build_plot_grid()
    frames = _fem_gif_frames(fem, x_mesh, y_mesh, n_frames=n_frames, t_end=t_end)
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_heat_vanilla_gif(
    output_path: Path | str = FIGS_DIR / "heat_vanilla.gif",
    n_epochs: int = HEAT_NUM_EPOCHS,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    frame_duration_ms: int | None = None,
    show: bool = False,
) -> Path:
    """GIF эволюции поля базовой модели (после обучения только по data loss)."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    x_mesh, y_mesh, x_flat, y_flat = _build_plot_grid()
    sensor_tensors = _prepare_sensor_data(fem)
    model = _train_vanilla_model(*sensor_tensors, n_epochs=n_epochs)
    frames = _model_gif_frames(
        model,
        "Базовая модель",
        x_mesh,
        y_mesh,
        x_flat,
        y_flat,
        n_frames=n_frames,
        t_end=t_end,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_heat_pinn_gif(
    output_path: Path | str = FIGS_DIR / "heat_pinn.gif",
    n_epochs: int = HEAT_NUM_EPOCHS,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    frame_duration_ms: int | None = None,
    show: bool = False,
) -> Path:
    """GIF эволюции поля PINN (после обучения data + PDE + IC + BC)."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    x_mesh, y_mesh, x_flat, y_flat = _build_plot_grid()
    sensor_tensors = _prepare_sensor_data(fem)
    model = _train_pinn_model(*sensor_tensors, n_epochs=n_epochs)
    frames = _model_gif_frames(
        model,
        "PINN",
        x_mesh,
        y_mesh,
        x_flat,
        y_flat,
        n_frames=n_frames,
        t_end=t_end,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_heat_prob_gif(
    output_path: Path | str = FIGS_DIR / "heat_prob.gif",
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    frame_duration_ms: int | None = None,
    show: bool = False,
) -> Path:
    """GIF расположения проб на пластине (только текущий набор замеров)."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    sensor_by_time = _build_sensor_data_by_time(fem)
    frames = _probe_gif_frames(sensor_by_time, n_frames=n_frames, t_end=t_end)
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_heat_colorbar_png(
    output_path: Path | str = FIGS_DIR / "heat_colorbar.png",
    dpi: int = HEAT_GIF_DPI,
    show: bool = False,
) -> Path:
    """PNG общей шкалы температур (ширина как у heat_*, высота 1/4)."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    _save_horizontal_colorbar_png(
        output_path,
        "jet",
        HEAT_T_VMIN,
        HEAT_T_VMAX,
        "T, °C",
        dpi=dpi,
        ticks=[20, 30, 40, 50, 60, 70],
    )
    if show:
        Image.open(output_path).show()
    return output_path


def plot_heat_err_fem_gif(
    output_path: Path | str = FIGS_DIR / "heat_err_fem.gif",
    n_epochs: int = HEAT_NUM_EPOCHS,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    frame_duration_ms: int | None = None,
    show: bool = False,
) -> Path:
    """GIF |МКЭ − МКЭ| относительно эталона (нулевая ошибка)."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    x_mesh, y_mesh, _, _ = _build_plot_grid()
    err_vmax = _get_err_vmax(fem, n_epochs=n_epochs)
    frames = _error_gif_frames(
        fem,
        lambda t_val: fem.solution(x_mesh, y_mesh, t_val),
        "Ошибка МКЭ",
        x_mesh,
        y_mesh,
        err_vmax,
        n_frames=n_frames,
        t_end=t_end,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_heat_err_vanilla_gif(
    output_path: Path | str = FIGS_DIR / "heat_err_vanilla.gif",
    n_epochs: int = HEAT_NUM_EPOCHS,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    frame_duration_ms: int | None = None,
    show: bool = False,
) -> Path:
    """GIF |базовая модель − МКЭ|."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    x_mesh, y_mesh, x_flat, y_flat = _build_plot_grid()
    err_vmax = _get_err_vmax(fem, n_epochs=n_epochs)
    vanilla_model = _get_vanilla_model(fem, n_epochs=n_epochs)
    frames = _error_gif_frames(
        fem,
        lambda t_val: _predict_field(
            vanilla_model, x_mesh, y_mesh, x_flat, y_flat, t_val
        ),
        "Ошибка базовой модели",
        x_mesh,
        y_mesh,
        err_vmax,
        n_frames=n_frames,
        t_end=t_end,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_heat_err_pinn_gif(
    output_path: Path | str = FIGS_DIR / "heat_err_pinn.gif",
    n_epochs: int = HEAT_NUM_EPOCHS,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    frame_duration_ms: int | None = None,
    show: bool = False,
) -> Path:
    """GIF |PINN − МКЭ|."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    x_mesh, y_mesh, x_flat, y_flat = _build_plot_grid()
    err_vmax = _get_err_vmax(fem, n_epochs=n_epochs)
    pinn_model = _get_pinn_model(fem, n_epochs=n_epochs)
    frames = _error_gif_frames(
        fem,
        lambda t_val: _predict_field(pinn_model, x_mesh, y_mesh, x_flat, y_flat, t_val),
        "Ошибка PINN",
        x_mesh,
        y_mesh,
        err_vmax,
        n_frames=n_frames,
        t_end=t_end,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_heat_err_colorbar_png(
    output_path: Path | str = FIGS_DIR / "heat_err_colorbar.png",
    n_epochs: int = HEAT_NUM_EPOCHS,
    dpi: int = HEAT_GIF_DPI,
    show: bool = False,
) -> Path:
    """PNG общей шкалы |ошибки| (ширина как у heat_*, высота 1/4)."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    err_vmax = _get_err_vmax(fem, n_epochs=n_epochs)
    _save_horizontal_colorbar_png(
        output_path,
        HEAT_ERR_CMAP,
        HEAT_ERR_VMIN,
        err_vmax,
        "|Ошибка|, °C",
        dpi=dpi,
        ticks=_error_colorbar_ticks(err_vmax),
    )

    if show:
        Image.open(output_path).show()
    return output_path


def plot_heat_mean_err_fem_gif(
    output_path: Path | str = FIGS_DIR / "heat_mean_err_fem.gif",
    n_epochs: int = HEAT_NUM_EPOCHS,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    frame_duration_ms: int | None = None,
    show: bool = False,
) -> Path:
    """GIF средней |МКЭ − МКЭ| по времени."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    curves, y_max = _get_mean_error_curves(fem, n_epochs=n_epochs)
    times, mean_errs = curves["fem"]
    frames = _mean_error_gif_frames(
        times,
        mean_errs,
        "Средняя ошибка МКЭ",
        y_max,
        n_frames=n_frames,
        t_end=t_end,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_heat_mean_err_vanilla_gif(
    output_path: Path | str = FIGS_DIR / "heat_mean_err_vanilla.gif",
    n_epochs: int = HEAT_NUM_EPOCHS,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    frame_duration_ms: int | None = None,
    show: bool = False,
) -> Path:
    """GIF средней |базовая модель − МКЭ| по времени."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    curves, y_max = _get_mean_error_curves(fem, n_epochs=n_epochs)
    times, mean_errs = curves["vanilla"]
    frames = _mean_error_gif_frames(
        times,
        mean_errs,
        "Средняя ошибка базовой модели",
        y_max,
        n_frames=n_frames,
        t_end=t_end,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


def plot_heat_mean_err_pinn_gif(
    output_path: Path | str = FIGS_DIR / "heat_mean_err_pinn.gif",
    n_epochs: int = HEAT_NUM_EPOCHS,
    n_frames: int = HEAT_FEM_N_FRAMES,
    t_end: float = max(T_PLOT_VALUES),
    frame_duration_ms: int | None = None,
    show: bool = False,
) -> Path:
    """GIF средней |PINN − МКЭ| по времени."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    fem = _get_fem_reference()
    curves, y_max = _get_mean_error_curves(fem, n_epochs=n_epochs)
    times, mean_errs = curves["pinn"]
    frames = _mean_error_gif_frames(
        times,
        mean_errs,
        "Средняя ошибка PINN",
        y_max,
        n_frames=n_frames,
        t_end=t_end,
    )
    _save_gif(frames, output_path, frame_duration_ms=frame_duration_ms)

    if show:
        frames[0].show()
    return output_path


PLOTS: dict[str, Callable[..., Path]] = {
    "heat_fem_gif": plot_heat_fem_gif,
    "heat_vanilla_gif": plot_heat_vanilla_gif,
    "heat_pinn_gif": plot_heat_pinn_gif,
    "heat_prob_gif": plot_heat_prob_gif,
    "heat_colorbar_png": plot_heat_colorbar_png,
    "heat_err_fem_gif": plot_heat_err_fem_gif,
    "heat_err_vanilla_gif": plot_heat_err_vanilla_gif,
    "heat_err_pinn_gif": plot_heat_err_pinn_gif,
    "heat_err_colorbar_png": plot_heat_err_colorbar_png,
    "heat_mean_err_fem_gif": plot_heat_mean_err_fem_gif,
    "heat_mean_err_vanilla_gif": plot_heat_mean_err_vanilla_gif,
    "heat_mean_err_pinn_gif": plot_heat_mean_err_pinn_gif,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GIF и PNG для 2D теплопроводности")
    parser.add_argument(
        "plot",
        nargs="?",
        default="heat_fem_gif",
        choices=sorted(PLOTS),
        help="Имя анимации для генерации",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=HEAT_FEM_N_FRAMES)
    parser.add_argument("--frame-duration-ms", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    plot_fn = PLOTS[args.plot]
    kwargs: dict = {"show": args.show}
    if args.output is not None:
        kwargs["output_path"] = args.output
    if args.plot in {
        "heat_fem_gif",
        "heat_vanilla_gif",
        "heat_pinn_gif",
        "heat_prob_gif",
        "heat_err_fem_gif",
        "heat_err_vanilla_gif",
        "heat_err_pinn_gif",
        "heat_mean_err_fem_gif",
        "heat_mean_err_vanilla_gif",
        "heat_mean_err_pinn_gif",
    }:
        kwargs["n_frames"] = args.n_frames
    if args.plot in {
        "heat_vanilla_gif",
        "heat_pinn_gif",
        "heat_err_fem_gif",
        "heat_err_vanilla_gif",
        "heat_err_pinn_gif",
        "heat_err_colorbar_png",
        "heat_mean_err_fem_gif",
        "heat_mean_err_vanilla_gif",
        "heat_mean_err_pinn_gif",
    }:
        if args.train_epochs is not None:
            kwargs["n_epochs"] = args.train_epochs
    if args.frame_duration_ms is not None:
        kwargs["frame_duration_ms"] = args.frame_duration_ms

    result = plot_fn(**kwargs)
    print(f"Сохранено: {result}")

    if args.plot in {
        "heat_fem_gif",
        "heat_vanilla_gif",
        "heat_pinn_gif",
        "heat_prob_gif",
        "heat_err_fem_gif",
        "heat_err_vanilla_gif",
        "heat_err_pinn_gif",
        "heat_mean_err_fem_gif",
        "heat_mean_err_vanilla_gif",
        "heat_mean_err_pinn_gif",
    }:
        frame_duration_ms = args.frame_duration_ms
        if frame_duration_ms is None:
            frame_duration_ms = _gif_frame_duration_ms(args.n_frames)
        total_duration_s = (
            (args.n_frames - 1) * frame_duration_ms + HEAT_GIF_FINAL_PAUSE_MS
        ) / 1000
        suffix = ""
        if args.plot in {
            "heat_vanilla_gif",
            "heat_pinn_gif",
            "heat_err_fem_gif",
            "heat_err_vanilla_gif",
            "heat_err_pinn_gif",
            "heat_mean_err_fem_gif",
            "heat_mean_err_vanilla_gif",
            "heat_mean_err_pinn_gif",
        }:
            n_epochs = args.train_epochs if args.train_epochs is not None else HEAT_NUM_EPOCHS
            suffix = f", обучение: {n_epochs} эпох"
        print(f"Кадров: {args.n_frames}{suffix}, ~{total_duration_s:.1f} с")
        if args.plot in {
            "heat_mean_err_fem_gif",
            "heat_mean_err_vanilla_gif",
            "heat_mean_err_pinn_gif",
        }:
            n_epochs = args.train_epochs if args.train_epochs is not None else HEAT_NUM_EPOCHS
            _print_mean_error_integrals(n_epochs=n_epochs)
    elif args.plot == "heat_err_colorbar_png":
        n_epochs = args.train_epochs if args.train_epochs is not None else HEAT_NUM_EPOCHS
        fem = _get_fem_reference()
        err_vmax = _get_err_vmax(fem, n_epochs=n_epochs)
        print(f"Шкала ошибок: 0 … {err_vmax:.2f} °C")
