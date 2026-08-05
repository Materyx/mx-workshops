from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_ball_track(csv_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Загрузить траекторию: τ = t − t₀, y в метрах."""
    path = Path(csv_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV не найден: {path}")

    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    t = np.asarray(table["t"], dtype=np.float64)
    y_px = np.asarray(table["y"], dtype=np.float64)

    scale = 1.112 / 1920
    tau = (t - t[0]).astype(np.float32)
    y_m = (y_px * scale).astype(np.float32)
    return tau, y_m
