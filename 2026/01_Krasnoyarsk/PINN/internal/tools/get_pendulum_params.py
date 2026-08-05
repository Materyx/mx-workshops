from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

SOURCE_DIR = Path(__file__).resolve().parent.parent
PINN_DIR = SOURCE_DIR.parent

DEFAULT_OMEGA0_FIT_POINTS = 10


@dataclass(frozen=True)
class PendulumParams:
    """Параметры аналитической модели и данные, нужные для pinn_pendulum."""

    beta: float
    omega0: float
    omega_dot: float
    c: float
    d: float
    rmse: float
    t_min: float
    t_max: float
    t_csv: np.ndarray
    theta_csv: np.ndarray
    pinn_beta: float
    pinn_omega: float
    pinn_theta0: float
    pinn_theta_dot0: float
    pinn_beta_phys: float
    pinn_omega0_sq: float

    @property
    def theta_dot0(self) -> float:
        return -self.beta * self.c + self.omega0 * self.d

    @property
    def t_analytical(self) -> np.ndarray:
        mask = (self.t_csv >= self.t_min) & (self.t_csv <= self.t_max)
        return self.t_csv[mask]

    @property
    def theta_analytical(self) -> np.ndarray:
        mask = (self.t_csv >= self.t_min) & (self.t_csv <= self.t_max)
        return self.theta_csv[mask]

    def evaluate(self, t_values: np.ndarray) -> np.ndarray:
        t_values = np.asarray(t_values, dtype=np.float64)
        phase = self.omega0 * t_values + 0.5 * self.omega_dot * t_values**2
        envelope = np.exp(-self.beta * t_values)
        return envelope * (self.c * np.cos(phase) + self.d * np.sin(phase))


def resolve_csv_path(csv_path: Path | str) -> Path:
    path = Path(csv_path).expanduser()
    if not path.is_absolute():
        path = (PINN_DIR / path).resolve()
    return path


def load_trajectory_csv(csv_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    path = resolve_csv_path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV не найден: {path}")

    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    t = np.asarray(table["t"], dtype=np.float64)
    theta = np.asarray(table["theta"], dtype=np.float64)
    return t, theta


def slice_time_window(
    t: np.ndarray,
    theta: np.ndarray,
    t_end: float,
    t_start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (t >= t_start) & (t <= t_end)
    return t[mask], theta[mask]


def sample_pendulum_data(
    params: PendulumParams,
    n_data: int,
    data_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Случайная подвыборка: равномерно по времени, θ — интерполяция CV-трека."""
    if n_data <= 0:
        raise ValueError(f"n_data должно быть > 0, получено {n_data}")
    if params.t_max <= params.t_min:
        raise ValueError(f"Некорректное окно [{params.t_min}, {params.t_max}]")
    if len(params.t_csv) < 2:
        raise ValueError("Нужно минимум 2 точки в CSV для интерполяции")

    rng = np.random.default_rng(data_seed)
    t_data = np.sort(rng.uniform(params.t_min, params.t_max, size=n_data))
    theta_data = np.interp(t_data, params.t_csv, params.theta_csv)
    return t_data.astype(np.float32), theta_data.astype(np.float32)


def rmse(reference: np.ndarray, predicted: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return float(np.sqrt(np.mean((reference - predicted) ** 2)))


def estimate_period(t: np.ndarray, theta: np.ndarray) -> float:
    dt = float(np.median(np.diff(t)))
    theta_centered = theta - np.mean(theta)
    freqs = np.fft.rfftfreq(len(t), dt)
    spectrum = np.abs(np.fft.rfft(theta_centered))
    if len(spectrum) <= 1:
        raise ValueError("Недостаточно точек для оценки периода")

    peak_index = 1 + int(np.argmax(spectrum[1:]))
    frequency = float(freqs[peak_index])
    if frequency <= 0:
        raise ValueError("Не удалось оценить частоту колебаний")

    return 1.0 / frequency


def get_beta_from_series(t: np.ndarray, theta: np.ndarray) -> float:
    period = estimate_period(t, theta)
    min_distance = max(
        3,
        int(round(0.35 * period / float(np.median(np.diff(t))))),
    )
    peaks, _ = find_peaks(np.abs(theta), distance=min_distance)
    if len(peaks) < 3:
        return 0.01

    t_peaks = t[peaks]
    amplitudes = np.abs(theta[peaks])
    valid = amplitudes > 0
    t_peaks = t_peaks[valid]
    amplitudes = amplitudes[valid]
    if len(t_peaks) < 2:
        return 0.01

    slope = float(np.polyfit(t_peaks, np.log(amplitudes), 1)[0])
    return max(-slope, 1e-4)


def damped_oscillation(
    t_values: np.ndarray,
    beta: float,
    omega: float,
    c: float,
    d: float,
) -> np.ndarray:
    envelope = np.exp(-beta * t_values)
    return envelope * (c * np.cos(omega * t_values) + d * np.sin(omega * t_values))


def _fit_analytical_guess(
    t: np.ndarray,
    theta: np.ndarray,
    n_fit: int = DEFAULT_OMEGA0_FIT_POINTS,
) -> tuple[float, float, float, float]:
    period = estimate_period(t, theta)
    omega_guess = 2.0 * np.pi / period
    beta_guess = max(get_beta_from_series(t, theta), 1e-4)
    c_guess = float(theta[0])
    theta_dot_guess = float(np.polyfit(t[:n_fit], theta[:n_fit], 1)[0])
    d_guess = (theta_dot_guess + beta_guess * c_guess) / omega_guess
    return beta_guess, omega_guess, c_guess, d_guess


def _fit_simple_damped_params(
    t: np.ndarray,
    theta: np.ndarray,
    n_fit: int = DEFAULT_OMEGA0_FIT_POINTS,
) -> tuple[float, float, float, float]:
    c_fixed = float(theta[0])
    beta_guess, omega_guess, _, d_guess = _fit_analytical_guess(t, theta, n_fit=n_fit)

    def damped_with_fixed_ic(t_values: np.ndarray, beta: float, omega: float, d: float) -> np.ndarray:
        return damped_oscillation(t_values, beta, omega, c_fixed, d)

    omega_fft = omega_guess
    beta_upper = min(2.5 * get_beta_from_series(t, theta), 0.5)
    params, _cov = curve_fit(
        damped_with_fixed_ic,
        t,
        theta,
        p0=[beta_guess, omega_guess, d_guess],
        bounds=(
            [0.0, max(0.5 * omega_fft, 1e-3), -np.inf],
            [beta_upper, min(1.5 * omega_fft, 20.0), np.inf],
        ),
        maxfev=50000,
    )
    beta_fit, omega_fit, d_fit = (float(v) for v in params)
    return beta_fit, omega_fit, c_fixed, d_fit


def _fit_chirp_params(
    t: np.ndarray,
    theta: np.ndarray,
    n_fit: int = DEFAULT_OMEGA0_FIT_POINTS,
) -> tuple[float, float, float, float, float, float]:
    c_fixed = float(theta[0])
    beta_guess, omega_guess, _, d_guess = _fit_analytical_guess(t, theta, n_fit=n_fit)

    def chirp_with_fixed_theta0(
        t_values: np.ndarray,
        beta: float,
        omega0: float,
        omega_dot: float,
        d: float,
    ) -> np.ndarray:
        phase = omega0 * t_values + 0.5 * omega_dot * t_values**2
        envelope = np.exp(-beta * t_values)
        return envelope * (c_fixed * np.cos(phase) + d * np.sin(phase))

    p0 = [beta_guess, omega_guess, 0.0, d_guess]
    bounds = (
        [0.0, max(0.5 * omega_guess, 1e-3), -0.1, -np.inf],
        [0.5, min(1.5 * omega_guess, 20.0), 0.1, np.inf],
    )

    fitted, _cov = curve_fit(chirp_with_fixed_theta0, t, theta, p0=p0, bounds=bounds, maxfev=100000)
    beta_fit, omega0_fit, omega_dot_fit, d_fit = (float(v) for v in fitted)
    prediction = chirp_with_fixed_theta0(t, beta_fit, omega0_fit, omega_dot_fit, d_fit)
    fit_rmse = rmse(theta, prediction)
    return beta_fit, omega0_fit, omega_dot_fit, c_fixed, d_fit, fit_rmse


def get_pendulum_params(
    csv_path: Path | str,
    t_max: float = 40.0,
    t_start: float = 0.0,
) -> PendulumParams:
    """Подобрать параметры аналитической модели и коэффициенты PINN по CSV."""
    t_csv, theta_csv = load_trajectory_csv(csv_path)
    t_window, theta_window = slice_time_window(t_csv, theta_csv, t_max, t_start)
    if len(t_window) < 20:
        raise ValueError(f"Слишком мало точек в окне [{t_start}, {t_max}] с")

    beta, omega0, omega_dot, c, d, fit_rmse = _fit_chirp_params(t_window, theta_window)

    pinn_beta, pinn_omega, pinn_c, pinn_d = _fit_simple_damped_params(t_csv, theta_csv)
    pinn_beta_phys = get_beta_from_series(t_csv, theta_csv)

    return PendulumParams(
        beta=beta,
        omega0=omega0,
        omega_dot=omega_dot,
        c=c,
        d=d,
        rmse=fit_rmse,
        t_min=t_start,
        t_max=t_max,
        t_csv=t_csv,
        theta_csv=theta_csv,
        pinn_beta=pinn_beta,
        pinn_omega=pinn_omega,
        pinn_theta0=pinn_c,
        pinn_theta_dot0=-pinn_beta * pinn_c + pinn_omega * pinn_d,
        pinn_beta_phys=pinn_beta_phys,
        pinn_omega0_sq=pinn_beta_phys**2 + pinn_omega**2,
    )


def format_params_for_constants(params: PendulumParams) -> str:
    return (
        f"beta={params.beta:.8f}\n"
        f"omega0={params.omega0:.8f}\n"
        f"omega_dot={params.omega_dot:.8f}\n"
        f"c={params.c:.4f}\n"
        f"d={params.d:.8f}\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Подбор параметров аналитической модели маятника")
    parser.add_argument(
        "csv",
        nargs="?",
        default=PINN_DIR / "Videos/Output/pendulums/pendulum_2_trajectory.csv",
    )
    parser.add_argument("--t-max", type=float, default=40.0, help="Правая граница окна подгонки, с")
    parser.add_argument("--t-start", type=float, default=0.0, help="Левая граница окна подгонки, с")
    args = parser.parse_args()

    params = get_pendulum_params(args.csv, t_max=args.t_max, t_start=args.t_start)
    print(f"Окно: t in [{args.t_start}, {args.t_max}] с")
    print(format_params_for_constants(params))
    print(f"RMSE = {params.rmse:.6f}")
    print(f"theta_dot(0) = {params.theta_dot0:.6f}")
