# Kaggle Week 5 — Object classification from geo-satellite snapshots

Проект для классификации спутниковых изображений на 20 классов. Метрика соревнования — Accuracy. Предстоит сравнить MLP, CNN и хотя бы одну pretrained-модель.

## Структура репозитория

```text
project/
├── data/
│   ├── raw/          # Исходные изображения и CSV
│   └── processed/    # Будущие производные данные
├── notebooks/
│   ├── 00_data_overview.ipynb
│   ├── 01_baseline_mlp.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_pretrained.ipynb
│   ├── 04_augmentations.ipynb
│   ├── 05_error_analysis.ipynb
│   └── 99_final_pipeline.ipynb
├── src/              # Будущие модули работы с данными, моделями и запусками
│   ├── models/ 
│   │   └── models.py
│   ├── training/ 
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── data.py
│   ├── transforms.py
│   └── utils.py
├── configs/          # Будущие конфигурации экспериментов
├── artifacts/
│   ├── models/       # Будущие сохранённые модели
│   ├── predictions/  # Будущие предсказания
│   └── logs/         # Будущие логи запусков
├── submissions/      # Будущие файлы для отправки в Kaggle
├── experiments.csv
├── findings.md
└── README.md
```

`project/` обозначает текущий корень репозитория.

Содержимое `data/raw/` не должно модифицироваться. Здесь хранятся исходные `train/`, `val/`, `test/`, `train.csv`, `val.csv`, `test.csv`, `classes.csv` и `sample_submission_example.csv`.

`experiments.csv` используется как журнал запусков, а `findings.md` хранит накопленные выводы и известные ограничения соревнования.

`notebooks/99_final_pipeline.ipynb` в будущем должен стать полностью воспроизводимым финальным решением.
