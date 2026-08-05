# PINN — Красноярск 2026

Материалы по Physics-Informed Neural Networks: демо для показа, практика для слушателей и служебные скрипты генерации.

## Структура

```
PINN/
├── assets/       # картинки для презентации (не перезаписываются примерами)
├── examples/     # готовые демо для показа слушателям
├── workshop/     # Jupyter-практика (пустые loss-ячейки + solutions/)
└── internal/     # служебное: трекинг видео, оценка параметров, генерация ассетов
```

| Каталог | Показывать? | Назначение |
|---------|-------------|------------|
| `examples/` | да | три полных PINN-скрипта |
| `workshop/` | да | 4 учебные задачи (2 физика + 2 химия) |
| `assets/` | да (картинки) | слайды / презентация |
| `internal/` | нет | CV, подбор параметров, пересборка фигур |

## Окружение

```bash
cd 2026/01_Krasnoyarsk/PINN
python3 -m pip install -r requirements.txt
# для трекинга видео дополнительно:
# python3 -m pip install opencv-python
```

Все скрипты считают пути от корня `PINN/` через `Path(__file__)` — **cwd не важен**.

## Examples

```bash
python3 examples/pinn_ball_trajectory.py
python3 examples/pinn_pendulum.py
python3 examples/pinn_heat_transfer_2d.py
```

- Данные: `examples/data/` (независимые копии, без связи с `internal/`)
- Новые картинки прогона: `examples/output/`

## Workshop

Ноутбуки с постановкой, аналитикой, данными и TODO под функции потерь:

1. `workshop/01_free_fall.ipynb` — свободное падение  
2. `workshop/02_harmonic_oscillator.ipynb` — гармонический осциллятор  
3. `workshop/03_first_order_kinetics.ipynb` — реакция 1-го порядка  
4. `workshop/04_reversible_reaction.ipynb` — обратимая реакция A ⇌ B  
5. `workshop/05_diffusion_reaction_2d.ipynb` — диффузия–реакция 2D (повышенная сложность)  
6. `workshop/06_diffusion_source_2d.ipynb` — диффузия с источником 2D (повышенная сложность)  

Задачи 1–4: `loss_data` / `loss_physics` / `loss_ic` (1 переменная).  
Задачи 5–6: `loss_data` / `loss_physics` / `loss_bc` (поле `c(x,y)`).

Эталоны для ведущего: `workshop/solutions/`.

Пересобрать синтетические csv (служебный скрипт):

```bash
python3 internal/tools/make_workshop_data.py
```

Открывать ноутбуки из каталога `workshop/` (или указать kernel с установленными зависимостями).

### Установка и настройка

Выполнить:

```bash
python3 -m pip install -r requirements.txt
python3 -m ipykernel install --user --name pinn --display-name "PINN (Python 3.11)"
```

## Internal (не для раздачи)

```bash
# CV-трекинг
python3 internal/video/track/track_pendulum.py
python3 internal/video/track/track_ball_throws.py

# Оценка параметров / гиперпараметров
python3 internal/tools/get_pendulum_params.py
python3 internal/tools/get_ball_params.py
python3 internal/tools/get_pendulum_hyperparams.py

# Синтетические данные для workshop/
python3 internal/tools/make_workshop_data.py

# Пересборка gif/png в assets/
python3 internal/tools/get_ball_graph.py
python3 internal/tools/get_pendulum_graph.py
python3 internal/tools/get_heat_graph.py
```

- Сырые видео: `internal/video/raw/`
- Треки (csv/mp4): `internal/video/tracks/`
- Презентационные ассеты: `assets/<topic>/`
