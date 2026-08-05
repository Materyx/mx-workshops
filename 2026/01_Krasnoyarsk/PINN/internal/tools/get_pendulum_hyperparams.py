from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from utils.get_pendulum_params import PendulumParams, get_pendulum_params, sample_pendulum_data

SOURCE_DIR = Path(__file__).resolve().parent.parent
PINN_DIR = SOURCE_DIR.parent

DEFAULT_SEARCH_METHOD = "bayesian"
DEFAULT_BO_TRIALS = 30
DEFAULT_SEARCH_TRIALS = 0
DEFAULT_COARSE_EPOCHS = 1500
DEFAULT_SEARCH_EPOCHS = 3000
DEFAULT_FINAL_EPOCHS = 7000
DEFAULT_REFINE_TOP_K = 1
DEFAULT_REFINE_PASSES = 4

DEFAULT_INITIAL_LAMBDAS = (10.0, 0.0, 20.0, 10.0)
LAMBDA_MIN = 0.1
LAMBDA_MAX = 50.0
REFINE_STEP_FACTOR = 2.0


@dataclass(frozen=True)
class PendulumHyperparams:
    lambda_data: float
    lambda_ode: float
    lambda_ic: float
    lambda_ic_vel: float
    rmse_analytical: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.lambda_data, self.lambda_ode, self.lambda_ic, self.lambda_ic_vel

    def __post_init__(self) -> None:
        for name, value in (
            ("lambda_data", self.lambda_data),
            ("lambda_ode", self.lambda_ode),
            ("lambda_ic", self.lambda_ic),
            ("lambda_ic_vel", self.lambda_ic_vel),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} должна быть > 0, получено {value:g}")


@dataclass(frozen=True)
class PendulumTrainConfig:
    csv_path: Path | str
    t_min: float = 0.0
    t_max: float = 60.0
    n_data: int = 200
    data_seed: int = 2
    n_hidden: int = 40
    lr: float = 0.01
    num_epochs: int = DEFAULT_SEARCH_EPOCHS
    n_phys: int = 500
    plot_points_per_period: int = 40
    train_seed: int = 0


class PINN(nn.Module):
    def __init__(self, n_hidden: int = 40, omega_feature: float = 1.0):
        super().__init__()
        self.omega_feature = float(omega_feature)
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        omega_t = self.omega_feature * t
        features = torch.cat([t, torch.sin(omega_t), torch.cos(omega_t)], dim=1)
        return self.net(features)


def _derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _clip_lambda(value: float, lambda_min: float = LAMBDA_MIN, lambda_max: float = LAMBDA_MAX) -> float:
    return float(np.clip(value, lambda_min, lambda_max))


def _sanitize_lambdas(
    lambdas: tuple[float, float, float, float],
    lambda_min: float = LAMBDA_MIN,
) -> tuple[float, float, float, float]:
    """Нулевые λ заменяются на lambda_min — PINN не может работать с λ=0."""
    return tuple(lambda_min if value <= 0.0 else float(value) for value in lambdas)


def _validate_lambdas(lambdas: tuple[float, float, float, float]) -> None:
    for index, value in enumerate(lambdas):
        if value <= 0.0:
            raise ValueError(f"λ[{index}] должна быть > 0, получено {value:g}")


def _random_lambdas(
    rng: random.Random,
    lambda_min: float = LAMBDA_MIN,
    lambda_max: float = LAMBDA_MAX,
) -> tuple[float, float, float, float]:
    log_min = np.log(lambda_min)
    log_max = np.log(lambda_max)
    return tuple(float(np.exp(rng.uniform(log_min, log_max))) for _ in range(4))


def _parse_initial_lambdas(text: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Ожидается 4 значения через запятую, например: 10,0,20,10")
    return tuple(float(part) for part in parts)


def _make_hyperparams(lambdas: tuple[float, float, float, float], rmse: float) -> PendulumHyperparams:
    _validate_lambdas(lambdas)
    return PendulumHyperparams(
        lambda_data=lambdas[0],
        lambda_ode=lambdas[1],
        lambda_ic=lambdas[2],
        lambda_ic_vel=lambdas[3],
        rmse_analytical=rmse,
    )


def _sample_training_data(
    params: PendulumParams,
    n_data: int,
    data_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_data, theta_data = sample_pendulum_data(params, n_data, data_seed)
    return (
        torch.tensor(t_data, dtype=torch.float32).view(-1, 1),
        torch.tensor(theta_data, dtype=torch.float32).view(-1, 1),
    )


def _evaluation_grid(params: PendulumParams, plot_points_per_period: int) -> np.ndarray:
    n_points = max(
        200,
        int(
            (params.t_max - params.t_min)
            * params.omega0
            / (2.0 * np.pi)
            * plot_points_per_period
        ),
    )
    return np.linspace(params.t_min, params.t_max, n_points, dtype=np.float32)


def evaluate_pinn_lambdas(
    lambdas: tuple[float, float, float, float],
    config: PendulumTrainConfig,
    params: PendulumParams | None = None,
    verbose: bool = False,
) -> float:
    """Обучить PINN с заданными λ и вернуть RMSE(PINN, аналитика) на окне."""
    _validate_lambdas(lambdas)
    lambda_data, lambda_ode, lambda_ic, lambda_ic_vel = lambdas
    params = params or get_pendulum_params(config.csv_path, t_max=config.t_max, t_start=config.t_min)

    _set_seed(config.train_seed)
    t_data, theta_data = _sample_training_data(params, config.n_data, config.data_seed)
    t_phys = torch.linspace(params.t_min, params.t_max, config.n_phys, dtype=torch.float32).view(-1, 1)

    model = PINN(n_hidden=config.n_hidden, omega_feature=params.pinn_omega)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    warmup_epochs = max(1, config.num_epochs // 2)
    model.train()
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()

        t0 = torch.zeros(1, 1, dtype=torch.float32)
        theta_pred_0 = model(t0)
        l_ic = (theta_pred_0 - params.pinn_theta0).pow(2).mean()

        t0_grad = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True)
        theta_pred_0_grad = model(t0_grad)
        theta_t_0 = _derivative(theta_pred_0_grad, t0_grad)
        l_ic_vel = (theta_t_0 - params.pinn_theta_dot0).pow(2).mean()

        theta_pred_data = model(t_data)
        l_data = torch.mean((theta_pred_data - theta_data) ** 2)

        if epoch < warmup_epochs:
            loss = lambda_data * l_data + lambda_ic * l_ic + lambda_ic_vel * l_ic_vel
        else:
            t_phys_grad = t_phys.clone().detach().requires_grad_(True)
            theta_pred_phys = model(t_phys_grad)
            theta_t = _derivative(theta_pred_phys, t_phys_grad)
            theta_tt = _derivative(theta_t, t_phys_grad)
            l_ode = torch.mean(
                (theta_tt + 2.0 * params.pinn_beta_phys * theta_t + params.pinn_omega0_sq * theta_pred_phys)
                ** 2
            )
            loss = (
                lambda_data * l_data
                + lambda_ode * l_ode
                + lambda_ic * l_ic
                + lambda_ic_vel * l_ic_vel
            )

        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % max(1, config.num_epochs // 5) == 0:
            print(f"  эпоха {epoch + 1}/{config.num_epochs}, loss={loss.item():.6f}")

    model.eval()
    t_eval = _evaluation_grid(params, config.plot_points_per_period)
    with torch.no_grad():
        theta_pred = model(torch.tensor(t_eval, dtype=torch.float32).view(-1, 1)).numpy().flatten()
    theta_analytical = params.evaluate(t_eval)
    return float(np.sqrt(np.mean((theta_pred - theta_analytical) ** 2)))


def _evaluate_trial(
    lambdas: tuple[float, float, float, float],
    config: PendulumTrainConfig,
    params: PendulumParams,
    label: str,
    verbose: bool,
) -> tuple[PendulumHyperparams, float]:
    if verbose:
        print(
            f"{label} λ=({lambdas[0]:g}, {lambdas[1]:g}, {lambdas[2]:g}, {lambdas[3]:g})"
        )
    rmse = evaluate_pinn_lambdas(lambdas, config, params=params, verbose=False)
    result = _make_hyperparams(lambdas, rmse)
    if verbose:
        print(f"  RMSE(PINN, аналитика) = {rmse:.5f}")
    return result, rmse


def _random_search(
    config: PendulumTrainConfig,
    params: PendulumParams,
    n_trials: int,
    search_seed: int,
    verbose: bool,
) -> list[tuple[PendulumHyperparams, float]]:
    rng = random.Random(search_seed)
    trials: list[tuple[PendulumHyperparams, float]] = []

    if verbose:
        print(
            f"Этап 1/2: случайный поиск в [{LAMBDA_MIN:g}, {LAMBDA_MAX:g}], "
            f"{n_trials} проб × {config.num_epochs} эпох"
        )

    for trial_index in range(n_trials):
        lambdas = _random_lambdas(rng)
        result, rmse = _evaluate_trial(
            lambdas,
            config,
            params,
            label=f"[{trial_index + 1}/{n_trials}]",
            verbose=verbose,
        )
        trials.append((result, rmse))

    trials.sort(key=lambda item: item[1])
    return trials


def _refine_lambdas_coordinate_descent(
    initial: tuple[float, float, float, float],
    config: PendulumTrainConfig,
    params: PendulumParams,
    max_passes: int = DEFAULT_REFINE_PASSES,
    step_factor: float = REFINE_STEP_FACTOR,
    verbose: bool = True,
) -> tuple[PendulumHyperparams, float]:
    best = list(initial)
    best_rmse = evaluate_pinn_lambdas(tuple(best), config, params=params, verbose=False)

    if verbose:
        print(
            f"  старт refine: λ=({best[0]:g}, {best[1]:g}, {best[2]:g}, {best[3]:g}), "
            f"RMSE={best_rmse:.5f}"
        )

    for pass_index in range(max_passes):
        improved_any = False
        for coordinate in range(4):
            for direction in (step_factor, 1.0 / step_factor):
                candidate = best.copy()
                candidate[coordinate] = _clip_lambda(candidate[coordinate] * direction)
                if candidate[coordinate] == best[coordinate]:
                    continue

                rmse = evaluate_pinn_lambdas(tuple(candidate), config, params=params, verbose=False)
                if rmse < best_rmse:
                    best = candidate
                    best_rmse = rmse
                    improved_any = True
                    if verbose:
                        print(
                            f"  pass {pass_index + 1}, λ[{coordinate}]×{direction:g}: "
                            f"λ=({best[0]:g}, {best[1]:g}, {best[2]:g}, {best[3]:g}), "
                            f"RMSE={best_rmse:.5f}"
                        )

        if not improved_any:
            break

    return _make_hyperparams(tuple(best), best_rmse), best_rmse


def _trial_to_hyperparams(trial: "optuna.trial.FrozenTrial") -> PendulumHyperparams:
    return PendulumHyperparams(
        lambda_data=trial.params["lambda_data"],
        lambda_ode=trial.params["lambda_ode"],
        lambda_ic=trial.params["lambda_ic"],
        lambda_ic_vel=trial.params["lambda_ic_vel"],
        rmse_analytical=float(trial.value),
    )


def bayesian_search_pendulum_hyperparams(
    config: PendulumTrainConfig,
    initial_lambdas: tuple[float, float, float, float] = DEFAULT_INITIAL_LAMBDAS,
    n_trials: int = DEFAULT_BO_TRIALS,
    search_seed: int = 0,
    startup_trials: int = 3,
    verbose: bool = True,
) -> list[tuple[PendulumHyperparams, float]]:
    """Байесовская оптимиза λ через Optuna (TPE-сampler), минимизация RMSE на всей кривой."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    params = get_pendulum_params(config.csv_path, t_max=config.t_max, t_start=config.t_min)
    start = _sanitize_lambdas(initial_lambdas)

    if verbose:
        print(
            f"Байесовская оптимизация (TPE): {n_trials} проб × {config.num_epochs} эпох, "
            f"λ ∈ [{LAMBDA_MIN:g}, {LAMBDA_MAX:g}] (log-uniform)"
        )
        print(
            f"Начальная проба: "
            f"({initial_lambdas[0]:g}, {initial_lambdas[1]:g}, {initial_lambdas[2]:g}, {initial_lambdas[3]:g}) "
            f"→ ({start[0]:g}, {start[1]:g}, {start[2]:g}, {start[3]:g})"
        )

    def objective(trial: optuna.Trial) -> float:
        lambdas = (
            trial.suggest_float("lambda_data", LAMBDA_MIN, LAMBDA_MAX, log=True),
            trial.suggest_float("lambda_ode", LAMBDA_MIN, LAMBDA_MAX, log=True),
            trial.suggest_float("lambda_ic", LAMBDA_MIN, LAMBDA_MAX, log=True),
            trial.suggest_float("lambda_ic_vel", LAMBDA_MIN, LAMBDA_MAX, log=True),
        )
        if verbose:
            print(
                f"[{trial.number + 1}/{n_trials}] "
                f"λ=({lambdas[0]:g}, {lambdas[1]:g}, {lambdas[2]:g}, {lambdas[3]:g})"
            )
        rmse = evaluate_pinn_lambdas(lambdas, config, params=params, verbose=False)
        if verbose:
            print(f"  RMSE(PINN, аналитика) = {rmse:.5f}")
        return rmse

    sampler = optuna.samplers.TPESampler(
        seed=search_seed,
        n_startup_trials=min(startup_trials, max(1, n_trials - 1)),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(
        {
            "lambda_data": start[0],
            "lambda_ode": start[1],
            "lambda_ic": start[2],
            "lambda_ic_vel": start[3],
        }
    )
    study.optimize(objective, n_trials=n_trials)

    results: list[tuple[PendulumHyperparams, float]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        hyperparams = _trial_to_hyperparams(trial)
        results.append((hyperparams, float(trial.value)))

    results.sort(key=lambda item: item[1])
    return results


def search_pendulum_hyperparams(
    config: PendulumTrainConfig,
    initial_lambdas: tuple[float, float, float, float] = DEFAULT_INITIAL_LAMBDAS,
    n_trials: int = DEFAULT_SEARCH_TRIALS,
    coarse_epochs: int = DEFAULT_COARSE_EPOCHS,
    refine_top_k: int = DEFAULT_REFINE_TOP_K,
    refine_passes: int = DEFAULT_REFINE_PASSES,
    search_seed: int = 0,
    verbose: bool = True,
) -> list[tuple[PendulumHyperparams, float]]:
    """Подбор λ: опциональный случайный поиск + coordinate descent от начального приближения."""
    params = get_pendulum_params(config.csv_path, t_max=config.t_max, t_start=config.t_min)
    all_trials: list[tuple[PendulumHyperparams, float]] = []

    if n_trials > 0:
        coarse_config = PendulumTrainConfig(
            csv_path=config.csv_path,
            t_min=config.t_min,
            t_max=config.t_max,
            n_data=config.n_data,
            data_seed=config.data_seed,
            n_hidden=config.n_hidden,
            lr=config.lr,
            num_epochs=coarse_epochs,
            n_phys=config.n_phys,
            plot_points_per_period=config.plot_points_per_period,
            train_seed=config.train_seed,
        )
        all_trials.extend(_random_search(coarse_config, params, n_trials, search_seed, verbose))

    start = _sanitize_lambdas(initial_lambdas)
    if verbose:
        print(
            f"\nУточнение от начального приближения "
            f"({initial_lambdas[0]:g}, {initial_lambdas[1]:g}, {initial_lambdas[2]:g}, {initial_lambdas[3]:g}) "
            f"→ ({start[0]:g}, {start[1]:g}, {start[2]:g}, {start[3]:g}), "
            f"{config.num_epochs} эпох"
        )

    refined, rmse = _refine_lambdas_coordinate_descent(
        start,
        config,
        params,
        max_passes=refine_passes,
        verbose=verbose,
    )
    all_trials.append((refined, rmse))

    if refine_top_k > 1 and n_trials > 0:
        if verbose:
            print(f"\nДополнительное уточнение top-{refine_top_k - 1} из случайного поиска")
        seen_starts = {start, refined.as_tuple()}
        for rank, (candidate, _) in enumerate(all_trials[:-1][: refine_top_k - 1], start=1):
            candidate_start = candidate.as_tuple()
            if candidate_start in seen_starts:
                continue
            seen_starts.add(candidate_start)
            if verbose:
                print(f"[refine extra {rank}]")
            extra, extra_rmse = _refine_lambdas_coordinate_descent(
                candidate_start,
                config,
                params,
                max_passes=refine_passes,
                verbose=verbose,
            )
            all_trials.append((extra, extra_rmse))

    all_trials.sort(key=lambda item: item[1])
    return all_trials


def get_pendulum_hyperparams(
    csv_path: Path | str,
    t_min: float = 0.0,
    t_max: float = 60.0,
    method: str = DEFAULT_SEARCH_METHOD,
    initial_lambdas: tuple[float, float, float, float] = DEFAULT_INITIAL_LAMBDAS,
    n_trials: int = DEFAULT_SEARCH_TRIALS,
    bo_trials: int = DEFAULT_BO_TRIALS,
    coarse_epochs: int = DEFAULT_COARSE_EPOCHS,
    search_epochs: int = DEFAULT_SEARCH_EPOCHS,
    final_epochs: int = DEFAULT_FINAL_EPOCHS,
    refine_top_k: int = DEFAULT_REFINE_TOP_K,
    refine_passes: int = DEFAULT_REFINE_PASSES,
    n_data: int = 200,
    data_seed: int = 2,
    train_seed: int = 0,
    search_seed: int = 0,
    verbose: bool = True,
) -> PendulumHyperparams:
    """Подобрать λ с минимальной ошибкой PINN относительно аналитики."""
    search_config = PendulumTrainConfig(
        csv_path=csv_path,
        t_min=t_min,
        t_max=t_max,
        n_data=n_data,
        data_seed=data_seed,
        num_epochs=search_epochs,
        train_seed=train_seed,
    )

    if method == "bayesian":
        trials = bayesian_search_pendulum_hyperparams(
            search_config,
            initial_lambdas=initial_lambdas,
            n_trials=bo_trials,
            search_seed=search_seed,
            verbose=verbose,
        )
        best = trials[0][0]
        if final_epochs > search_epochs:
            if verbose:
                print(f"\nФинальная переоценка лучших λ при {final_epochs} эпохах")
            final_config = PendulumTrainConfig(
                csv_path=csv_path,
                t_min=t_min,
                t_max=t_max,
                n_data=n_data,
                data_seed=data_seed,
                num_epochs=final_epochs,
                train_seed=train_seed,
            )
            params = get_pendulum_params(csv_path, t_max=t_max, t_start=t_min)
            final_rmse = evaluate_pinn_lambdas(best.as_tuple(), final_config, params=params)
            best = _make_hyperparams(best.as_tuple(), final_rmse)
            if verbose:
                print(f"  RMSE(PINN, аналитика) = {final_rmse:.5f}")
    elif method == "refine":
        final_config = PendulumTrainConfig(
            csv_path=csv_path,
            t_min=t_min,
            t_max=t_max,
            n_data=n_data,
            data_seed=data_seed,
            num_epochs=final_epochs if final_epochs else search_epochs,
            train_seed=train_seed,
        )
        trials = search_pendulum_hyperparams(
            final_config,
            initial_lambdas=initial_lambdas,
            n_trials=n_trials,
            coarse_epochs=coarse_epochs,
            refine_top_k=refine_top_k,
            refine_passes=refine_passes,
            search_seed=search_seed,
            verbose=verbose,
        )
        best = trials[0][0]
    else:
        raise ValueError(f"Неизвестный method={method!r}, ожидается 'bayesian' или 'refine'")
    if verbose:
        print("\nЛучшие гиперпараметры:")
        print(format_hyperparams_for_constants(best))
    return best


def format_hyperparams_for_constants(params: PendulumHyperparams) -> str:
    return (
        f"lambda_data = {params.lambda_data:g}\n"
        f"lambda_ode = {params.lambda_ode:g}\n"
        f"lambda_ic = {params.lambda_ic:g}\n"
        f"lambda_ic_vel = {params.lambda_ic_vel:g}\n"
        f"# RMSE(PINN, аналитика) = {params.rmse_analytical:.5f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Подбор λ для PINN маятника")
    parser.add_argument(
        "csv",
        nargs="?",
        default=PINN_DIR / "Videos/Output/pendulums/pendulum_2_trajectory.csv",
    )
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=60.0)
    parser.add_argument(
        "--method",
        choices=("bayesian", "refine"),
        default=DEFAULT_SEARCH_METHOD,
        help="bayesian — Optuna TPE; refine — coordinate descent от --initial",
    )
    parser.add_argument(
        "--initial",
        type=str,
        default=",".join(str(value) for value in DEFAULT_INITIAL_LAMBDAS),
        help="Начальное приближение λ_data,λ_ode,λ_ic,λ_ic_vel (0 → 0.1)",
    )
    parser.add_argument(
        "--bo-trials",
        type=int,
        default=DEFAULT_BO_TRIALS,
        help="Число проб байесовской оптимизации",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_SEARCH_TRIALS,
        help="Случайных проб для method=refine; 0 = только refine от --initial",
    )
    parser.add_argument("--coarse-epochs", type=int, default=DEFAULT_COARSE_EPOCHS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_SEARCH_EPOCHS)
    parser.add_argument("--final-epochs", type=int, default=DEFAULT_FINAL_EPOCHS)
    parser.add_argument("--refine-top-k", type=int, default=DEFAULT_REFINE_TOP_K)
    parser.add_argument("--refine-passes", type=int, default=DEFAULT_REFINE_PASSES)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--n-data", type=int, default=200)
    parser.add_argument("--data-seed", type=int, default=2)
    parser.add_argument("--train-seed", type=int, default=0)
    args = parser.parse_args()

    get_pendulum_hyperparams(
        args.csv,
        t_min=args.t_min,
        t_max=args.t_max,
        method=args.method,
        initial_lambdas=_parse_initial_lambdas(args.initial),
        n_trials=args.trials,
        bo_trials=args.bo_trials,
        coarse_epochs=args.coarse_epochs,
        search_epochs=args.epochs,
        final_epochs=args.final_epochs,
        refine_top_k=args.refine_top_k,
        refine_passes=args.refine_passes,
        n_data=args.n_data,
        data_seed=args.data_seed,
        train_seed=args.train_seed,
        search_seed=args.search_seed,
        verbose=True,
    )
