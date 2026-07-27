# mx-workshops

Материалы открытых семинаров [Materyx](https://materyx.ru): код, данные и визуализации.

## Семинары

| | |
|---|---|
| **2026 / Красноярск** | Physics-Informed Neural Networks (PINN) |

### PINN (Красноярск, 2026)

Практика по физико-информированным нейросетям на PyTorch:

- траектория мяча (`pinn_ball_trajectory.py`)
- маятник с данными с видео (`pinn_pendulum.py`)
- 2D теплоперенос (`pinn_heat_transfer_2d.py`)

Также есть скрипты трекинга объектов с видео (`Videos/Scripts/`) и готовые траектории в CSV.

```
2026/01_Krasnoyarsk/PINN/
├── Source/     # PINN-модели и utils
├── Videos/     # исходники, трекинг, CSV-траектории
└── Figs/       # результаты обучения
```

**Запуск:**

```bash
cd 2026/01_Krasnoyarsk/PINN/Source
pip install -r requirements.txt
python pinn_ball_trajectory.py
```

## Лицензия

MIT © Materyx
