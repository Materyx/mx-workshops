from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PINN_DIR = ROOT
ASSETS = ROOT / "assets"
TRACKS = ROOT / "internal" / "video" / "tracks"
RAW = ROOT / "internal" / "video" / "raw"

G = 9.8
EXCLUDE_TAIL = 3
DEFAULT_DATA_CSV = TRACKS / "ball_throws" / "good" / "track06.csv"
BALL_VIDEO = RAW / "ball_throws" / "ball_throws.mp4"
FRAME_WIDTH_PX = 1920
WIDTH_SEARCH_MIN_M = 0.5
WIDTH_SEARCH_MAX_M = 1.5
WIDTH_SEARCH_TOL_M = 1e-4


@dataclass(frozen=True)
class BallParams:
    """Константы и данные для pinn_ball_trajectory."""

    g: float
    y0: float
    v0: float
    tau_max: float
    tau_train: np.ndarray
    y_train: np.ndarray
    tau_all: np.ndarray
    y_all: np.ndarray
    tau_analytical_max: float

    def evaluate(self, tau_values: np.ndarray) -> np.ndarray:
        tau_values = np.asarray(tau_values, dtype=np.float64)
        return self.y0 + self.v0 * tau_values - 0.5 * self.g * tau_values**2


def _resolve_csv_path(csv_path: Path | str) -> Path:
    path = Path(csv_path).expanduser()
    if not path.is_absolute():
        path = (PINN_DIR / path).resolve()
    return path


def _load_csv(csv_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    path = _resolve_csv_path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV не найден: {path}")

    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.asarray(table["t"], dtype=np.float64), np.asarray(table["y"], dtype=np.float64)


def _frame_width_px() -> int:
    if not BALL_VIDEO.is_file():
        return FRAME_WIDTH_PX

    import cv2

    capture = cv2.VideoCapture(str(BALL_VIDEO))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture.release()
    return width or FRAME_WIDTH_PX


def _fit_y0_v0(tau: np.ndarray, y_m: np.ndarray, g: float = G) -> tuple[float, float]:
    k2 = -0.5 * g
    design = np.column_stack([np.ones_like(tau), tau])
    target = y_m - k2 * tau**2
    y0, v0 = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(y0), float(v0)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _rmse_for_width(
    tau_fit: np.ndarray,
    y_px_fit: np.ndarray,
    frame_width_m: float,
    width_px: int,
    g: float = G,
) -> float:
    y_m = y_px_fit * (frame_width_m / width_px)
    y0, v0 = _fit_y0_v0(tau_fit, y_m, g=g)
    y_pred = y0 + v0 * tau_fit - 0.5 * g * tau_fit**2
    return _rmse(y_m, y_pred)


def _calibrate_frame_width_m(
    tau_fit: np.ndarray,
    y_px_fit: np.ndarray,
    width_px: int,
    g: float = G,
) -> float:
    left, right = WIDTH_SEARCH_MIN_M, WIDTH_SEARCH_MAX_M
    while right - left > WIDTH_SEARCH_TOL_M:
        third = (right - left) / 3.0
        mid_left = left + third
        mid_right = right - third
        if _rmse_for_width(tau_fit, y_px_fit, mid_left, width_px, g=g) <= _rmse_for_width(
            tau_fit, y_px_fit, mid_right, width_px, g=g
        ):
            right = mid_right
        else:
            left = mid_left
    return 0.5 * (left + right)


def get_ball_params(
    csv_path: Path | str = DEFAULT_DATA_CSV,
    g: float = G,
    exclude_tail: int = EXCLUDE_TAIL,
) -> BallParams:
    t_csv, y_px = _load_csv(csv_path)
    n_train = len(t_csv) - exclude_tail
    if n_train < 2:
        raise ValueError("Недостаточно точек для подгонки")

    width_px = _frame_width_px()
    frame_width_m = _calibrate_frame_width_m(
        t_csv[:n_train] - t_csv[0],
        y_px[:n_train],
        width_px,
        g=g,
    )
    scale = frame_width_m / width_px

    tau_all = (t_csv - t_csv[0]).astype(np.float32)
    y_all = (y_px * scale).astype(np.float32)
    tau_train = tau_all[:n_train]
    y_train = y_all[:n_train]
    y0, v0 = _fit_y0_v0(tau_train.astype(np.float64), y_train.astype(np.float64), g=g)

    return BallParams(
        g=float(g),
        y0=y0,
        v0=v0,
        tau_max=float(tau_all[-1]),
        tau_train=tau_train,
        y_train=y_train,
        tau_all=tau_all,
        y_all=y_all,
        tau_analytical_max=float(tau_train[-1]),
    )
