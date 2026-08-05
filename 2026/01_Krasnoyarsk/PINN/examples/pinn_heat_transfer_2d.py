import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

FIGS_DIR = Path(__file__).resolve().parent.parent / 'Figs'
FIGS_DIR.mkdir(parents=True, exist_ok=True)
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------
# 1. Генерация синтетических данных
# ---------------------------------------

# Геометрия пластины
w = 0.25   # ширина, м (ось x)
h = 0.1    # высота, м (ось y)

# Физические параметры алюминия
k = 237.0       # теплопроводность, Вт/(м·К)
rho = 2700.0    # плотность, кг/м³
c = 900.0       # удельная теплоёмкость, Дж/(кг·К)
alpha = k / (rho * c)  # коэффициент температуропроводности, м²/с

# Температурные условия
T_init = 20.0   # начальная температура, °C
T_left = 20.0   # температура на левой границе, °C
T_right = 70.0  # температура на правой границе, °C

# Временной интервал
t_min = 0.0
t_max = 400.0
t_plot_values = [1, 40, 100, 400, 800]

# ---------------------------------------------------------
# Эталонное решение через МКЭ (P1-треугольники, scipy.sparse)
# ---------------------------------------------------------
def build_fem_mesh(nx_cells, ny_cells):
    """Структурированная сетка узлов и треугольных элементов."""
    nx_nodes = nx_cells + 1
    ny_nodes = ny_cells + 1
    x_nodes = np.linspace(0.0, w, nx_nodes)
    y_nodes = np.linspace(0.0, h, ny_nodes)

    def node_id(i, j):
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

def triangle_matrices(coords):
    """Локальные матрицы жёсткости и масс для P1-треугольника."""
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    B = np.vstack([b, c]) / (2.0 * area)

    K_local = alpha * area * (B.T @ B)
    M_local = (area / 12.0) * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ])
    return K_local, M_local

def assemble_fem_system(nodes, elements):
    """Глобальные разреженные матрицы K и M."""
    n_nodes = nodes.shape[0]
    K = sparse.lil_matrix((n_nodes, n_nodes))
    M = sparse.lil_matrix((n_nodes, n_nodes))

    for elem in elements:
        coords = nodes[elem]
        K_loc, M_loc = triangle_matrices(coords)
        for a in range(3):
            for b in range(3):
                K[elem[a], elem[b]] += K_loc[a, b]
                M[elem[a], elem[b]] += M_loc[a, b]

    return K.tocsr(), M.tocsr()

def apply_dirichlet(A, b, dirichlet_map):
    """Подстановка граничных значений Дирихле в СЛАУ."""
    A = A.tolil()
    for node, value in dirichlet_map.items():
        A.rows[node] = [node]
        A.data[node] = [1.0]
        b[node] = value
    return A.tocsr(), b

def run_fem_solver():
    """Неявный Эйлер: dT/dt = alpha * laplacian(T) с заданными НУ и ГУ."""
    nx_cells, ny_cells = 50, 20
    dt = 2.0
    nodes, elements, x_nodes, y_nodes, left_nodes, right_nodes = build_fem_mesh(
        nx_cells, ny_cells
    )
    K, M = assemble_fem_system(nodes, elements)

    dirichlet = {node: T_left for node in left_nodes}
    dirichlet.update({node: T_right for node in right_nodes})

    n_nodes = nodes.shape[0]
    T = np.full(n_nodes, T_init)

    snapshots = {0.0: T.copy()}
    t_current = 0.0
    next_snapshot_idx = 0
    snapshot_targets = sorted(t for t in t_plot_values if t > 0)

    A_base = (M / dt + K).tocsc()
    M_dt = (M / dt).tocsc()

    while t_current < t_max - 1e-9:
        t_current += dt
        b = M_dt @ T
        A, b = apply_dirichlet(A_base.copy(), b.copy(), dirichlet)
        T = spsolve(A, b)

        if (next_snapshot_idx < len(snapshot_targets) and
                t_current >= snapshot_targets[next_snapshot_idx] - 1e-9):
            t_snap = snapshot_targets[next_snapshot_idx]
            snapshots[t_snap] = T.copy()
            next_snapshot_idx += 1

    ny_nodes = ny_cells + 1
    nx_nodes = nx_cells + 1
    interpolators = {}
    for t_snap, T_nodes in snapshots.items():
        T_grid = T_nodes.reshape(ny_nodes, nx_nodes)
        interpolators[t_snap] = RegularGridInterpolator(
            (y_nodes, x_nodes), T_grid,
            bounds_error=False, fill_value=None
        )

    return np.array(sorted(snapshots.keys())), interpolators

def fem_solution(x, y, t):
    """
    Интерполяция МКЭ-решения в произвольных (x, y, t).
    Пространство — bilinear (RegularGridInterpolator), время — линейно.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    shape = np.broadcast_shapes(x.shape, y.shape, t.shape)
    x_b = np.broadcast_to(x, shape).ravel()
    y_b = np.broadcast_to(y, shape).ravel()
    t_b = np.broadcast_to(t, shape).ravel()

    pts = np.column_stack([y_b, x_b])
    T_snap = np.array([
        fem_interpolators[t_snap](pts) for t_snap in fem_times
    ])

    result = np.zeros(len(x_b))
    for i in range(len(x_b)):
        result[i] = np.interp(t_b[i], fem_times, T_snap[:, i])

    return result.reshape(shape)

print("Расчёт эталонного МКЭ-решения...")
fem_times, fem_interpolators = run_fem_solver()
print(f"МКЭ готов: снимки при t = {list(fem_interpolators.keys())} с")

# Генерируем редкие точки-«датчики» для каждого момента времени отдельно
# (стабильные позиции через фиксированный seed на каждый t)
SENSOR_SEED_BASE = 4
N_sensors = 15
t_sensors = np.array(t_plot_values, dtype=np.float64)
noise_level = 2.0  # °C

sensor_data_by_time = {}
x_data_list, y_data_list, t_data_list, T_data_list = [], [], [], []

for t_val in t_sensors:
    rng = np.random.default_rng(SENSOR_SEED_BASE + int(t_val))
    sx = rng.uniform(0.0, w, N_sensors)
    sy = rng.uniform(0.0, h, N_sensors)
    T_exact = fem_solution(sx, sy, t_val)
    T_noisy = T_exact + noise_level * rng.standard_normal(N_sensors)
    sensor_data_by_time[float(t_val)] = {'x': sx, 'y': sy, 'T': T_noisy}
    x_data_list.extend(sx)
    y_data_list.extend(sy)
    t_data_list.extend([t_val] * N_sensors)
    T_data_list.extend(T_noisy)

# Разворачиваем в плоский набор (x, y, t, T) для обучения
x_data = np.array(x_data_list)
y_data = np.array(y_data_list)
t_data = np.array(t_data_list)
T_data = np.array(T_data_list)

# Преобразуем в тензоры PyTorch
x_data_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)
y_data_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)
t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
T_data_tensor = torch.tensor(T_data, dtype=torch.float32).view(-1, 1)

# --------------------------------------------------------------
# 2. Определяем небольшую полносвязную нейросеть для T(x, y, t)
# --------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, n_hidden=32):
        super(PINN, self).__init__()
        # Простой MLP с 3 скрытыми слоями
        self.net = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x, y, t):
        """
        Прямой проход: входы (x, y, t) нормируются к [0, 1].
        """
        x_norm = x / w
        y_norm = y / h
        t_norm = t / t_max
        inputs = torch.cat([x_norm, y_norm, t_norm], dim=1)
        return self.net(inputs)

# Создаём экземпляры моделей
model = PINN(n_hidden=32)
vanilla_model = PINN(n_hidden=32)

# -----------------------------------------------------
# 3. Вспомогательная функция для автодифференцирования
# -----------------------------------------------------
def derivative(y, x):
    """
    Вычисляет dy/dx с помощью autograd в PyTorch.
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
#    (1) Потеря по данным (подгонка под зашумлённые измерения датчиков)
#    (2) Потеря по УЧП: dT/dt = alpha * (d2T/dx2 + d2T/dy2)
#    (3) Потеря по начальному условию: T(x, y, 0) = 20
#    (4) Потеря по граничным условиям: Дирихле и Нейман

def sample_interior_points(n_points):
    """Случайные точки внутри области (x, y, t)."""
    x = torch.rand(n_points, 1) * w
    y = torch.rand(n_points, 1) * h
    t = torch.rand(n_points, 1) * t_max
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    return x, y, t

def sample_initial_points(n_points):
    """Случайные точки при t = 0."""
    x = torch.rand(n_points, 1) * w
    y = torch.rand(n_points, 1) * h
    t = torch.zeros(n_points, 1)
    return x, y, t

def sample_boundary_points(n_points):
    """Случайные точки на границах пластины."""
    n_each = n_points // 4
    # Для ГУ Дирихле избегаем t=0, где конфликтуют НУ и ГУ
    t_bc_min = 1.0

    # Левая граница: x = 0, T = 20
    x_left = torch.zeros(n_each, 1)
    y_left = torch.rand(n_each, 1) * h
    t_left = t_bc_min + torch.rand(n_each, 1) * (t_max - t_bc_min)

    # Правая граница: x = w, T = 70
    x_right = torch.full((n_each, 1), w)
    y_right = torch.rand(n_each, 1) * h
    t_right = t_bc_min + torch.rand(n_each, 1) * (t_max - t_bc_min)

    # Нижняя граница: y = 0, dT/dy = 0
    x_bottom = torch.rand(n_each, 1) * w
    y_bottom = torch.zeros(n_each, 1)
    t_bottom = torch.rand(n_each, 1) * t_max
    y_bottom.requires_grad_(True)

    # Верхняя граница: y = h, dT/dy = 0
    x_top = torch.rand(n_each, 1) * w
    y_top = torch.full((n_each, 1), h)
    t_top = torch.rand(n_each, 1) * t_max
    y_top.requires_grad_(True)

    return (x_left, y_left, t_left,
            x_right, y_right, t_right,
            x_bottom, y_bottom, t_bottom,
            x_top, y_top, t_top)

def physics_loss(model, n_points=2000):
    """
    Невязка уравнения теплопроводности:
    dT/dt - alpha * (d2T/dx2 + d2T/dy2) = 0
    """
    x, y, t = sample_interior_points(n_points)

    T_pred = model(x, y, t)
    dT_dt = derivative(T_pred, t)
    dT_dx = derivative(T_pred, x)
    d2T_dx2 = derivative(dT_dx, x)
    dT_dy = derivative(T_pred, y)
    d2T_dy2 = derivative(dT_dy, y)

    residual = dT_dt - alpha * (d2T_dx2 + d2T_dy2)
    return torch.mean(residual ** 2)

def initial_condition_loss(model, n_points=500):
    """
    Начальное условие: T(x, y, 0) = 20 °C.
    """
    x, y, t = sample_initial_points(n_points)
    T_pred = model(x, y, t)
    return torch.mean((T_pred - T_init) ** 2)

def boundary_condition_loss(model, n_points=400):
    """
    Граничные условия:
    - слева:  T(0, y, t) = 20
    - справа: T(w, y, t) = 70
    - снизу и сверху: dT/dy = 0
    """
    (x_left, y_left, t_left,
     x_right, y_right, t_right,
     x_bottom, y_bottom, t_bottom,
     x_top, y_top, t_top) = sample_boundary_points(n_points)

    # Дирихле: левая граница
    T_left_pred = model(x_left, y_left, t_left)
    loss_left = torch.mean((T_left_pred - T_left) ** 2)

    # Дирихле: правая граница
    T_right_pred = model(x_right, y_right, t_right)
    loss_right = torch.mean((T_right_pred - T_right) ** 2)

    # Нейман: нижняя граница
    T_bottom_pred = model(x_bottom, y_bottom, t_bottom)
    dT_dy_bottom = derivative(T_bottom_pred, y_bottom)
    loss_bottom = torch.mean(dT_dy_bottom ** 2)

    # Нейман: верхняя граница
    T_top_pred = model(x_top, y_top, t_top)
    dT_dy_top = derivative(T_top_pred, y_top)
    loss_top = torch.mean(dT_dy_top ** 2)

    return loss_left + loss_right + loss_bottom + loss_top

def data_loss(model, x_data, y_data, t_data, T_data):
    """
    MSE между предсказанной температурой и зашумлёнными измерениями датчиков.
    """
    T_pred = model(x_data, y_data, t_data)
    return torch.mean((T_pred - T_data) ** 2)

# ---------------------------------------
# 5. Настройка обучения
# ---------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Гиперпараметры — веса компонентов функции потерь
lambda_data = 2.0
lambda_pde  = 2.0
lambda_ic   = 2.0
lambda_bc   = 2.0

num_epochs = 6000
print_every = 200

# ---------------------------------------
# 6. Цикл обучения PINN
# ---------------------------------------
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()

    l_data = data_loss(model, x_data_tensor, y_data_tensor, t_data_tensor, T_data_tensor)
    l_pde  = physics_loss(model)
    l_ic   = initial_condition_loss(model)
    l_bc   = boundary_condition_loss(model)

    loss = (lambda_data * l_data + lambda_pde * l_pde +
            lambda_ic * l_ic + lambda_bc * l_bc)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % print_every == 0:
        print(f"PINN — эпоха {epoch+1}/{num_epochs}, "
              f"Общая потеря = {loss.item():.6f}, "
              f"Потеря по данным = {l_data.item():.6f}, "
              f"Потеря по УЧП = {l_pde.item():.6f}, "
              f"Потеря по НУ = {l_ic.item():.6f}, "
              f"Потеря по ГУ = {l_bc.item():.6f}")

# ---------------------------------------
# 6b. Цикл обучения базовой модели (только данные, без физики)
# ---------------------------------------
vanilla_optimizer = torch.optim.Adam(vanilla_model.parameters(), lr=1e-3)
vanilla_model.train()
for epoch in range(num_epochs):
    vanilla_optimizer.zero_grad()
    l_data = data_loss(
        vanilla_model, x_data_tensor, y_data_tensor, t_data_tensor, T_data_tensor
    )
    l_data.backward()
    vanilla_optimizer.step()

    if (epoch + 1) % print_every == 0:
        print(f"Базовая модель — эпоха {epoch+1}/{num_epochs}, "
              f"Потеря по данным = {l_data.item():.6f}")

# ---------------------------------------
# 7. Оценка обученных моделей
# ---------------------------------------
model.eval()
vanilla_model.eval()

# Мелкая равномерная сетка для визуализации
nx, ny = 200, 80
x_grid = np.linspace(0, w, nx)
y_grid = np.linspace(0, h, ny)
X, Y = np.meshgrid(x_grid, y_grid)

x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1)

T_vmin, T_vmax = 20.0, 70.0
results = []

for t_val in t_plot_values:
    T_true = fem_solution(X, Y, t_val)
    t_flat = torch.full_like(x_flat, t_val)
    with torch.no_grad():
        T_vanilla = vanilla_model(x_flat, y_flat, t_flat).numpy().reshape(X.shape)
        T_pinn = model(x_flat, y_flat, t_flat).numpy().reshape(X.shape)

    T_err_vanilla = np.abs(T_vanilla - T_true)
    T_err_pinn = np.abs(T_pinn - T_true)

    sd = sensor_data_by_time[float(t_val)]
    results.append({
        't_val': t_val,
        'T_true': T_true,
        'T_vanilla': T_vanilla,
        'T_pinn': T_pinn,
        'T_err_vanilla': T_err_vanilla,
        'T_err_pinn': T_err_pinn,
        'sensor_x': sd['x'],
        'sensor_y': sd['y'],
        'sensor_T': sd['T'],
    })

err_vmax = max(
    max(r['T_err_vanilla'].max() for r in results),
    max(r['T_err_pinn'].max() for r in results),
)
err_cmap = 'terrain'

n_rows = len(t_plot_values)
fig = plt.figure(figsize=(30, 3.2 * n_rows + 2))
gs = GridSpec(
    n_rows + 1, 6, figure=fig,
    height_ratios=[1] * n_rows + [0.05],
    hspace=0.12, wspace=0.35,
    top=0.94, bottom=0.06, left=0.04, right=0.98
)
axes = np.array([[fig.add_subplot(gs[row, col]) for col in range(6)] for row in range(n_rows)])

column_titles = [
    'МКЭ', 'Данные', 'Базовая модель', 'PINN',
    'Ошибка базовой', 'Ошибка PINN'
]

for row, res in enumerate(results):
    t_val = res['t_val']

    pcm_fem = axes[row, 0].pcolormesh(
        X, Y, res['T_true'], cmap='jet', vmin=T_vmin, vmax=T_vmax, shading='auto'
    )
    axes[row, 0].set_ylabel(f't = {t_val} с', fontsize=11)
    if row == 0:
        axes[row, 0].set_title(column_titles[0], fontsize=12, pad=4)

    axes[row, 1].scatter(
        res['sensor_x'], res['sensor_y'],
        c=res['sensor_T'], cmap='jet', vmin=T_vmin, vmax=T_vmax,
        s=80, edgecolors='k', linewidths=0.5
    )
    if row == 0:
        axes[row, 1].set_title(column_titles[1], fontsize=12, pad=4)

    pcm_vanilla = axes[row, 2].pcolormesh(
        X, Y, res['T_vanilla'], cmap='jet', vmin=T_vmin, vmax=T_vmax, shading='auto'
    )
    if row == 0:
        axes[row, 2].set_title(column_titles[2], fontsize=12, pad=4)

    pcm_pinn = axes[row, 3].pcolormesh(
        X, Y, res['T_pinn'], cmap='jet', vmin=T_vmin, vmax=T_vmax, shading='auto'
    )
    if row == 0:
        axes[row, 3].set_title(column_titles[3], fontsize=12, pad=4)

    pcm_err_vanilla = axes[row, 4].pcolormesh(
        X, Y, res['T_err_vanilla'], cmap=err_cmap, vmin=0, vmax=err_vmax, shading='auto'
    )
    if row == 0:
        axes[row, 4].set_title(column_titles[4], fontsize=12, pad=4)

    pcm_err_pinn = axes[row, 5].pcolormesh(
        X, Y, res['T_err_pinn'], cmap=err_cmap, vmin=0, vmax=err_vmax, shading='auto'
    )
    if row == 0:
        axes[row, 5].set_title(column_titles[5], fontsize=12, pad=4)

    for col in range(6):
        axes[row, col].set_aspect('equal', adjustable='box')
        axes[row, col].set_xlim(0, w)
        axes[row, col].set_ylim(0, h)

# Общие горизонтальные шкалы под всеми графиками
cax_temp = fig.add_subplot(gs[n_rows, 0:4])
cax_err = fig.add_subplot(gs[n_rows, 4:6])

cbar_temp = fig.colorbar(pcm_pinn, cax=cax_temp, orientation='horizontal')
cbar_temp.set_label('T, °C')

cbar_err = fig.colorbar(pcm_err_pinn, cax=cax_err, orientation='horizontal')
cbar_err.set_label('|Ошибка|, °C')

fig.suptitle(
    'PINN для 2D теплопроводности в алюминиевой пластине',
    fontsize=14, y=0.98
)
plt.savefig(FIGS_DIR / 'pinn_heat_transfer_2d_result.png', dpi=150)
plt.show()

# При необходимости сохраняем модель в файл
# torch.save(model.state_dict(), 'pinn_heat_transfer_2d_model.pth')