import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.load_ball_track import load_ball_track

FIGS_DIR = Path(__file__).resolve().parent.parent / "Figs" / "ball_trajectory"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 1. Аналитическое решение
# ---------------------------------------
g = 9.8  # Ускорение свободного падения, м/с^2
y0 = 0.294  # Начальная высота, м
v0 = 2.373  # Начальная скорость, м/с

# Аналитическое решение для траектории мяча
def analytical_solution(t):
    t = np.asarray(t, dtype=np.float64)
    return y0 + v0 * t - 0.5 * g * t**2

# ---------------------------------------
# 2. Зашумлённые данные
# ---------------------------------------
t, y = load_ball_track("Videos/Output/ball_throws/bad/track06.csv")

t_data_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
y_data_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# --------------------------------------------------------------
# 3. Определяем небольшую полносвязную нейросеть для h(t)
# --------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, n_hidden=20): # n_hidden - количество скрытых нейронов
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden), # Первый слой: входной слой для времени t
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden), # Второй слой: скрытый слой
            nn.Tanh(),
            nn.Linear(n_hidden, 1), # Третий слой: выходной слой для высоты h(t)
        )

    def forward(self, t):
        return self.net(t)

model = PINN(n_hidden=20) # Создаем модель PINN с 20 скрытыми нейронами

# Производная функции h(t) по времени t
# y - высота h(t)
# x - время t
# grad_outputs - градиенты высоты h(t) по времени t
# create_graph - создать граф для обратного распространения ошибки
def derivative(y, x): 
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

# ----------------------------------------------
# 4. Компоненты функции потерь (PINN)
# ----------------------------------------------
# У нас есть:
#    (1) Потеря по данным (подгонка под зашумлённые измерения датчиков)
#    (2) Потеря по ОДУ: dh/dt = v0 - g * t
#    (3) Потеря по начальному условию: h(0) = y0

# Потеря по ОДУ
def physics_loss(model, t):
    t = t.clone().detach().requires_grad_(True)
    h_pred = model(t)
    dh_dt_pred = derivative(h_pred, t)
    dh_dt_true = v0 - g * t
    return torch.mean((dh_dt_pred - dh_dt_true) ** 2)

# Потеря по начальному условию
def initial_condition_loss(model):
    t0 = torch.zeros(1, 1, dtype=torch.float32)
    return (model(t0) - y0).pow(2).mean()

# Потеря по данным
def data_loss(model, t, y):
    return torch.mean((model(t) - y) ** 2)


# ---------------------------------------
# 5. Настройка обучения
# ---------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Оптимизатор Adam с learning rate = 0.01

lambda_data = 1.0 # Вес для потери по данным
lambda_ode = 1.0 # Вес для потери по ОДУ
lambda_ic = 1.0 # Вес для потери по начальному условию

num_epochs = 4000 # Количество эпох
print_every = 200 # Печатать каждые 200 эпох

# ---------------------------------------
# 6. Цикл обучения
# ---------------------------------------
model.train() # Устанавливаем модель в режим обучения

# Цикл обучения
for epoch in range(num_epochs):
    optimizer.zero_grad()

    l_data = data_loss(model, t_data_tensor, y_data_tensor)
    l_ode = physics_loss(model, t_data_tensor)
    l_ic = initial_condition_loss(model)

    loss = lambda_data * l_data + lambda_ode * l_ode + lambda_ic * l_ic
    loss.backward()
    optimizer.step()

    if (epoch + 1) % print_every == 0:
        print(
            f"Эпоха {epoch + 1}/{num_epochs}, "
            f"Общая потеря = {loss.item():.6f}, "
            f"Потеря по данным = {l_data.item():.6f}, "
            f"Потеря по ОДУ = {l_ode.item():.6f}, "
            f"Потеря по начальному условию = {l_ic.item():.6f}"
        )

# ---------------------------------------
# 7. Оценка обученной модели
# ---------------------------------------
model.eval()

t_plot = np.linspace(0.0, float(t[-1]), 100).reshape(-1, 1).astype(np.float32)
t_plot_tensor = torch.tensor(t_plot, requires_grad=True)

h_pred_plot = model(t_plot_tensor).detach().numpy()

t_analytical = np.linspace(0.0, float(t[-1]), 100)
h_true_plot = analytical_solution(t_analytical)
valid = h_true_plot >= 0

plt.figure(figsize=(8, 5))
plt.scatter(t, y, color="red", label="Зашумленные данные")
plt.plot(t_analytical[valid], h_true_plot[valid], "k--", label="Точное решение")
plt.plot(t_plot, h_pred_plot, "b", label="Предсказание модели")
plt.xlabel("t, с")
plt.ylabel("y, м")
plt.legend()
plt.title("PINN для траектории броска мяча")
plt.grid(True)
plt.savefig(FIGS_DIR / f"pinn_ball_trajectory_result_lambda_{lambda_data}_{lambda_ode}_{lambda_ic}.png")
plt.show()
