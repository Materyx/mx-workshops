import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
OUTPUT = ROOT / "examples" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "examples"))
from lib.pendulum_data import get_pendulum_params, sample_pendulum_data

# ---------------------------------------
# 0. Fourier-признаки
# ---------------------------------------
# True  — вход (t, sin(ωt), cos(ωt))
# False — вход только t
is_fourier_features = True
train_seed = 42
torch.manual_seed(train_seed)
np.random.seed(train_seed)

# ---------------------------------------
# 1. Загрузка данных (из track_pendulum CSV)
# ---------------------------------------

# Путь к траектории
DATA_CSV = DATA / "pendulum" / "pendulum_2_trajectory.csv"

# Временное окно, с
t_min, t_max = 0.0, 60.0

# Физические параметры PINN (оценка по CV-треку)
PARAMS = get_pendulum_params(DATA_CSV, t_max=t_max, t_start=t_min)
beta_phys = PARAMS.pinn_beta_phys      # коэффициент затухания β
omega0_sq = PARAMS.pinn_omega0_sq      # ω₀² = β² + ω²
theta0 = PARAMS.pinn_theta0
theta_dot0 = PARAMS.pinn_theta_dot0
omega = PARAMS.pinn_omega              # частота для Fourier-признаков
t_csv = PARAMS.t_csv
theta_csv = PARAMS.theta_csv

# Аналитическое решение: θ(t) = e^(-βt)(c·cos φ + d·sin φ), φ = ω₀t + ½ω̇t²
def analytical_solution(t):
    return PARAMS.evaluate(t)

# Случайная подвыборка обучающих точек
N_data = 50
data_seed = 42
t_data, theta_data = sample_pendulum_data(PARAMS, N_data, data_seed)

# Преобразуем в тензоры PyTorch
t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
theta_data_tensor = torch.tensor(theta_data, dtype=torch.float32).view(-1, 1)

# --------------------------------------------------------------
# 2. Определяем полносвязную нейросеть для θ(t)
# --------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, n_hidden=20, omega_feature=1.0, use_fourier_features=True):
        super(PINN, self).__init__()
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

    def features(self, t):
        if self.use_fourier_features:
            omega_t = self.omega_feature * t
            return torch.cat([t, torch.sin(omega_t), torch.cos(omega_t)], dim=1)
        return t

    def forward(self, t):
        """
        Прямой проход: вход формы (batch_size, 1) -> выход формы (batch_size, 1)
        """
        return self.net(self.features(t))

# Создаём экземпляр модели
model = PINN(n_hidden=60, omega_feature=omega, use_fourier_features=is_fourier_features)

# -----------------------------------------------------
# 3. Вспомогательная функция для автодифференцирования
# -----------------------------------------------------
def derivative(y, x):
    """
    Вычисляет dy/dx с помощью autograd в PyTorch.
    y и x должны быть тензорами, при этом для x нужно requires_grad=True.
    """
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]

# ----------------------------------------------
# 4. Определяем компоненты функции потерь (PINN)
# ----------------------------------------------
# У нас есть:
#    (1) Потеря по данным (подвыборка CV-трека)
#    (2) Потеря по ОДУ: θ'' + 2β θ' + ω₀² θ = 0
#    (3) Потеря по начальному условию: θ(0) = θ₀
#    (4) Потеря по начальной скорости: dθ/dt(0) = θ̇₀

def physics_loss(model, t_phys):
    """
    Невязка линейного затухающего осциллятора:
    θ'' + 2β θ' + ω₀² θ = 0
    """
    t_phys = t_phys.clone().detach().requires_grad_(True)
    theta_pred = model(t_phys)
    theta_t = derivative(theta_pred, t_phys)
    theta_tt = derivative(theta_t, t_phys)
    residual = theta_tt + 2.0 * beta_phys * theta_t + omega0_sq * theta_pred
    return torch.mean(residual ** 2)

def initial_condition_loss(model):
    """
    Обеспечиваем выполнение условия θ(0) = θ₀.
    """
    t0 = torch.zeros(1, 1, dtype=torch.float32)
    theta_pred_0 = model(t0)
    return (theta_pred_0 - theta0).pow(2).mean()

def initial_velocity_loss(model):
    """
    Обеспечиваем выполнение условия dθ/dt(0).
    """
    t0 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True)
    theta_pred_0 = model(t0)
    theta_t_0 = derivative(theta_pred_0, t0)
    return (theta_t_0 - theta_dot0).pow(2).mean()

def data_loss(model, t_data, theta_data):
    """
    MSE между предсказанной θ(t_i) и подвыборкой CV-трека.
    """
    theta_pred = model(t_data)
    return torch.mean((theta_pred - theta_data) ** 2)

# ---------------------------------------
# 5. Настройка обучения
# ---------------------------------------
# Adam — оптимизатор; lr — скорость обучения
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Гиперпараметры — веса компонентов функции потерь
lambda_data = 48.0
lambda_ode = 0.39
lambda_ic = 0.23
lambda_ic_vel = 0.35

num_epochs = 10000
print_every = 200
N_phys = 500
warmup_epochs = num_epochs // 2
PLOT_POINTS_PER_PERIOD = 40

# Точки для штрафа по ОДУ
t_phys = torch.linspace(t_min, t_max, N_phys, dtype=torch.float32).view(-1, 1)

# ---------------------------------------
# 6. Цикл обучения
# ---------------------------------------
model.train()

for epoch in range(num_epochs):
    optimizer.zero_grad()

    l_data = data_loss(model, t_data_tensor, theta_data_tensor)
    l_ode = physics_loss(model, t_phys)
    l_ic = initial_condition_loss(model)
    l_ic_vel = initial_velocity_loss(model)

    # Первая половина — без ODE (warmup), вторая — полная функция потерь
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

    if (epoch + 1) % print_every == 0:
        print(
            f"Эпоха {epoch + 1}/{num_epochs}, "
            f"Общая потеря = {loss.item():.6f}, "
            f"Потеря по данным = {l_data.item():.6f}, "
            f"Потеря по ОДУ = {l_ode.item():.6f}, "
            f"Потеря по θ(0) = {l_ic.item():.6f}, "
            f"Потеря по dθ/dt(0) = {l_ic_vel.item():.6f}"
        )

# ---------------------------------------
# 7. Оценка обученной модели
# ---------------------------------------
model.eval()

# Сетки для графика
n_pinn_points = max(400, int((t_max - t_min) * omega / (2.0 * np.pi) * PLOT_POINTS_PER_PERIOD))
t_plot = np.linspace(t_min, t_max, n_pinn_points, dtype=np.float32)
t_plot_tensor = torch.tensor(t_plot, dtype=torch.float32).view(-1, 1)
theta_pred_plot = model(t_plot_tensor).detach().numpy().flatten()

n_analytical_points = max(
    200,
    int((t_max - t_min) * PARAMS.omega0 / (2.0 * np.pi) * PLOT_POINTS_PER_PERIOD),
)
t_analytical_plot = np.linspace(t_min, t_max, n_analytical_points, dtype=np.float32)
theta_analytical_plot = analytical_solution(t_analytical_plot)
window_mask = (t_csv >= t_min) & (t_csv <= t_max)

# Имя файла с коэффициентами λ
def _format_lambda_coeff(value):
    formatted = f"{value:.1f}"
    return formatted.rstrip('0').rstrip('.')

lambda_suffix = ",".join(
    _format_lambda_coeff(value)
    for value in (lambda_data, lambda_ode, lambda_ic, lambda_ic_vel)
)
fourier_tag = 'fourier' if is_fourier_features else 'plain'
result_fig_path = OUTPUT / f'pinn_pendulum_result_{fourier_tag}_lambda[{lambda_suffix}].png'

plt.figure(figsize=(10, 5))
plt.plot(
    t_csv[window_mask],
    theta_csv[window_mask],
    color='0.35',
    linewidth=0.8,
    label='Численное решение (CV)',
)
plt.scatter(
    t_data,
    theta_data,
    color='red',
    s=18,
    label='Данные (выборка)',
)
plt.plot(
    t_analytical_plot,
    theta_analytical_plot,
    'k--',
    linewidth=1.5,
    label='Аналитическое решение',
)
plt.plot(
    t_plot,
    theta_pred_plot,
    'b',
    linewidth=1.2,
    label='Предсказание модели',
)
plt.xlim(t_min, t_max)
plt.xlabel('t')
plt.ylabel('θ(t) [rad]')
plt.legend()
plt.title(
    f"PINN для затухающих колебаний маятника "
    f"({t_min:.0f}–{t_max:.0f} с; "
    rf"$\lambda_{{\mathrm{{data}}}}={lambda_data:g}$, "
    rf"$\lambda_{{\mathrm{{ode}}}}={lambda_ode:g}$, "
    rf"$\lambda_{{\mathrm{{ic}}}}={lambda_ic:g}$, "
    rf"$\lambda_{{\mathrm{{ic,vel}}}}={lambda_ic_vel:g}$)"
)
plt.grid(True)
plt.savefig(result_fig_path)
plt.show()
