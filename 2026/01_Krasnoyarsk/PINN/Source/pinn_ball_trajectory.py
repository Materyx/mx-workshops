import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGS_DIR = Path(__file__).resolve().parent.parent / 'Figs'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 1. Генерация синтетических данных
# ---------------------------------------
# Физические параметры
g = 9.8        # ускорение свободного падения
h0 = 1.0       # начальная высота
v0 = 10.0      # начальная скорость

# Точное (аналитическое) решение h(t) = h0 + v0*t - 0.5*g*t^2
def true_solution(t):
    return h0 + v0*t - 0.5*g*(t**2)

# Генерируем набор моментов времени
t_min, t_max = 0.0, 2.0
N_data = 10
t_data = np.linspace(t_min, t_max, N_data)

# Генерируем синтетические "экспериментальные" высоты с шумом
np.random.seed(42)
noise_level = 0.7 # 70% шума
h_data_exact = true_solution(t_data)
h_data_noisy = h_data_exact + noise_level*np.random.randn(N_data)

# Преобразуем в тензоры PyTorch
t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
h_data_tensor = torch.tensor(h_data_noisy, dtype=torch.float32).view(-1, 1)

# --------------------------------------------------------------
# 2. Определяем небольшую полносвязную нейросеть для h(t)
# --------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, n_hidden=20): # n_hidden - количество нейронов в скрытом слое (по умолчанию 20)
        super(PINN, self).__init__()
        # Простой многослойный персептрон (MLP) с 2 скрытыми слоями
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, t):
        """
        Прямой проход: вход формы (batch_size, 1) -> выход формы (batch_size, 1)
        """
        return self.net(t)

# Создаём экземпляр модели
model = PINN(n_hidden=20)

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
        create_graph=True
    )[0]
    
# ----------------------------------------------
# 4. Определяем компоненты функции потерь (PINN)
# ----------------------------------------------
# У нас есть:
#    (1) Потеря по данным (подгонка под зашумлённые данные)
#    (2) Потеря по ОДУ: dh/dt = v0 - g * t
#    (3) Потеря по начальному условию: h(0) = h0

def physics_loss(model, t):
    """
    Сравниваем d(h_pred)/dt с известным выражением (v0 - g t).
    """
    # Для работы autograd тензор t должен иметь requires_grad = True
    t.requires_grad_(True)

    h_pred = model(t)
    dh_dt_pred = derivative(h_pred, t)

    # Для каждого t истинное уравнение: dh/dt = v0 - g * t
    dh_dt_true = v0 - g * t

    # Потеря по ОДУ: среднеквадратичная ошибка между предсказанным и истинным dh/dt
    loss_ode = torch.mean((dh_dt_pred - dh_dt_true)**2)
    return loss_ode

def initial_condition_loss(model):
    """
    Обеспечиваем выполнение условия h(0) = h0.
    """
    t0 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False) 
    h0_pred = model(t0)
    
    loss_ic = (h0_pred - h0).pow(2).mean() # Потеря по начальному условию: среднеквадратичная ошибка между предсказанным и истинным h(0)
    return loss_ic

def data_loss(model, t_data, h_data):
    """
    Среднеквадратичная ошибка (MSE) между предсказанными h(t_i)
    и зашумлёнными измерениями h_data.
    """
    h_pred = model(t_data) # Предсказываем значения h(t) для всех t_data
    
    # Потеря по данным: среднеквадратичная ошибка между предсказанными и зашумлёнными измерениями
    loss_data = torch.mean((h_pred - h_data)**2)
    return loss_data

# ---------------------------------------
# 5. Настройка обучения
# ---------------------------------------
# Определяем оптимизатор. 
# Adam - это алгоритм оптимизации, который используется для обновления весов нейросети
# lr - learning rate, т.е. скорость обучения
# model.parameters() - это все параметры нейросети, которые нужно оптимизировать
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Гиперпараметры — веса компонентов функции потерь
lambda_data = 2.0
lambda_ode  = 2.0
lambda_ic   = 2.0

# num_epochs - количество эпох обучения
# print_every - частота вывода прогресса
num_epochs = 4000
print_every = 200

# ---------------------------------------
# 6. Цикл обучения
# ---------------------------------------
# Переводим модель в режим обучения
model.train()

# Цикл обучения
for epoch in range(num_epochs):
    # Обнуляем градиенты
    optimizer.zero_grad()

    # Вычисляем потери
    l_data = data_loss(model, t_data_tensor, h_data_tensor)
    l_ode  = physics_loss(model, t_data_tensor)
    l_ic   = initial_condition_loss(model)

    # Суммарная потеря
    loss = lambda_data * l_data + lambda_ode * l_ode + lambda_ic * l_ic

    # Обратное распространение
    loss.backward()
    optimizer.step()

    # Вывод прогресса
    if (epoch+1) % print_every == 0:
        print(f"Эпоха {epoch+1}/{num_epochs}, "
              f"Общая потеря = {loss.item():.6f}, "
              f"Потеря по данным = {l_data.item():.6f}, "
              f"Потеря по ОДУ = {l_ode.item():.6f}, "
              f"Потеря по начальному условию = {l_ic.item():.6f}")
        
# ---------------------------------------
# 7. Оценка обученной модели
# ---------------------------------------
# Переводим модель в режим оценки
model.eval()

# Генерируем точки для оценки
t_plot = np.linspace(t_min, t_max, 100).reshape(-1, 1).astype(np.float32)
t_plot_tensor = torch.tensor(t_plot, requires_grad=True)

# Предсказываем значения h(t) для всех t_plot
h_pred_plot = model(t_plot_tensor).detach().numpy()

# Точное решение (для сравнения)
h_true_plot = true_solution(t_plot)

# Строим графики результатов
plt.figure(figsize=(8, 5))
plt.scatter(t_data, h_data_noisy, color='red', label='Зашумлённые данные')
plt.plot(t_plot, h_true_plot, 'k--', label='Точное решение')
plt.plot(t_plot, h_pred_plot, 'b', label='Предсказание PINN')
plt.xlabel('t')
plt.ylabel('h(t)')
plt.legend()
plt.title('PINN для траектории мяча')
plt.grid(True)
plt.savefig(FIGS_DIR / 'pinn_ball_trajectory_result.png')
plt.show()

# При необходимости сохраняем модель в файл
# torch.save(model.state_dict(), 'pinn_ball_trajectory_model.pth')
