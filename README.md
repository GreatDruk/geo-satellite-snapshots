# Kaggle Week 5 — Object classification from geo-satellite snapshots

Классификация спутниковых изображений на 20 классов; метрика — Accuracy. Train содержит 4999 изображений, val — 699, test — 8299. Ограничения соревнования и накопленные выводы хранятся в `findings.md`.

## Структура

```text
project/
├── data/raw/                 # Неизменяемые исходные изображения и CSV
├── data/processed/           # Производные данные будущих экспериментов
├── notebooks/                # Запуски и будущий анализ
├── src/
│   ├── data.py               # CSV, Dataset, DataLoader; порядок классов из classes.csv
│   ├── transforms.py         # Resize, фиксированная нормализация, train-аугментации
│   ├── models/models.py      # Сборка MLP/CNN по списку слоёв
│   ├── training/train.py     # Одна эпоха обучения
│   ├── training/evaluate.py  # Validation loss и accuracy
│   ├── training/predict.py   # Предсказания и экспорт submission
│   ├── experiment.py         # Полный цикл эксперимента
│   └── utils.py              # Seed, журнал, графики
├── configs/
│   ├── baseline_mlp.json
│   └── baseline_cnn.json
├── artifacts/
│   ├── models/<exp_id>/
│   ├── predictions/<exp_id>/
│   └── logs/<exp_id>/
├── submissions/
├── tests/test_pipeline.py    # Проверки на синтетических данных
├── experiments.csv           # Журнал завершённых запусков
├── findings.md               # Накопленные выводы
└── README.md
```

`project/` — текущий корень репозитория. `data/raw/` не должен модифицироваться.

## Запуск

Откройте `notebooks/01_baseline_mlp.ipynb` или `02_baseline_cnn.ipynb` с Python-ядром, в котором доступны зависимости из `pyproject.toml`. В notebook достаточно импорта, уникального имени эксперимента и пути к конфигу:

```python
from src.experiment import run_experiment

result = run_experiment("configs/baseline_cnn.json", exp_id="cnn_001")
```

В обоих notebooks уже есть короткая ячейка, добавляющая корень проекта в путь импорта. Пути к конфигам и данным вычисляются относительно корня, поэтому запуск из `notebooks/` также работает.

Из терминала в корне:

```bash
python -m src.experiment configs/baseline_cnn.json --exp-id cnn_001
```

Конфиги — стартовые настройки, а не результаты или оптимальные гиперпараметры. Для нового опыта скопируйте JSON либо загрузите его через `load_config`, измените словарь и передайте в `run_experiment`. Функция сохраняет фактически использованный конфиг.

## Настройка конфигурации

- `seed` фиксирует Python, NumPy, PyTorch и seed генераторов DataLoader/workers. `training.deterministic=true` включает строгие детерминированные алгоритмы. Повторяемость ожидается в одинаковом программном и аппаратном окружении; при deterministic_warn_only=false неподдерживаемая детерминированная операция завершит запуск ошибкой; при true выдаст предупреждение.
- `data.image_size`: число для квадратного входа или `[height, width]`. `batch_size`, `num_workers`, `pin_memory` настраивают загрузку. По умолчанию `num_workers=0` удобно для Windows notebooks.
- `data.normalization.mean/std`: фиксированные RGB-значения; код не подбирает статистики на test. По умолчанию это `[0.5, 0.5, 0.5]`.
- `data.augmentation`: пустой объект отключает аугментации. Поддерживаются отражения, повороты, изменения цвета/gamma, шум, blur, JPEG и снижение детализации. Аугментации применяются только к train; описание параметров и сравнимых CNN-конфигов — в [docs/augmentations.md](docs/augmentations.md).
- `model.layers`: последовательность слоёв. Поддерживаются `conv2d`, `max_pool2d`, `adaptive_avg_pool2d`, `batch_norm2d`, `flatten`, `linear`, `dropout`, `relu`, `leaky_relu`, `gelu`, `silu`, `sigmoid`, `tanh`. Параметры соответствуют PyTorch; размер входа выводится автоматически. Последний слой — `linear` с `out_features: "num_classes"`. Выход — logits, Softmax перед CrossEntropyLoss не нужен.
- `training`: `scheduler` (опционально; `null` отключает), `epochs`, `log_every` (каждая k-я эпоха; первая и последняя также печатаются), `device` (`auto`, `cpu`, `cuda`), `deterministic`, `optimizer`. Поддерживаются Adam, AdamW, SGD с параметрами PyTorch.
- `notes`: комментарий для журнала.

CNN использует adaptive pooling, поэтому размер входа можно менять без ручного пересчёта Linear. MLP получает размер Flatten из `image_size`.

## Результаты запуска

После успешного опыта автоматически сохраняются:

- `artifacts/models/<exp_id>/best.pt` и `last.pt`: веса, состояния optimizer и scheduler, конфиг, эпоха и порядок классов. Best выбирается по максимальной val accuracy; при равенстве сохраняется более ранняя эпоха.
- `artifacts/logs/<exp_id>/config.json`, `metadata.json`, `history.csv`, `training.log`, `summary.json`, `learning_curves.png`: настройки, окружение, метрики каждой эпохи, сообщения и графики loss/accuracy. В notebook графики также выводятся автоматически.
- `artifacts/predictions/<exp_id>/val.csv`: ID, предсказанный label и true_label лучшей модели.
- Одна строка в `experiments.csv`: лучшая val accuracy в диапазоне 0–1, время цикла обучения в минутах, seed и параметры. `public_score` остаётся пустым до получения реального результата Kaggle.

`result` содержит лучшую модель, DataFrame истории, лучшую эпоху, accuracy и пути. Восстановить сохранённую модель можно через `load_checkpoint` из `src.experiment`. Checkpoints не реализуют точное продолжение прерванного обучения: для этого дополнительно нужны состояния RNG и загрузчиков.

Каждый запуск должен иметь уникальный `exp_id`. Существующий журнал и артефакты не перезаписываются. Запуски с общим журналом выполняйте последовательно. Если запуск прервался, сохранённые метрики/checkpoints и сообщение об ошибке остаются в его директориях; новая строка успешного опыта не добавляется. Следующий запуск получает новый ID.

## Test и будущие модели

По умолчанию test не загружается. Явно включить inference лучшей модели и создание submission можно так:

```python
result = run_experiment("configs/baseline_cnn.json", exp_id="cnn_002", predict_test=True)
```

Будут дополнительно созданы `artifacts/predictions/cnn_002/test.csv` и `submissions/cnn_002.csv`, с ID в порядке sample submission. Test не участвует в обучении, выборе checkpoint, fitting preprocessing, feature selection, representations или clustering. Отправка в Kaggle вручную.

Обучение, оценка и сохранение не зависят от конкретной архитектуры. Для будущей модели предусмотрен аргумент `model_factory`: функция `(model_config, image_size, num_classes) -> torch.nn.Module`, возвращающая logits формы `[batch, num_classes]`. Та же фабрика нужна при `load_checkpoint`. Встроены torchvision ResNet18/34/50/101/152 и режимы transfer learning; подробности в docs/transfer_learning.md. Разрешены AlexNet, VGG, ResNet и Inception; Transformers запрещены. Нормализацию, размер входа, заморозку слоёв и единственный выход logits будущая фабрика/конфиг должны задавать явно.

`findings.md` хранит только известные сведения и накопленные выводы. `99_final_pipeline.ipynb` воспроизводит выбранный на val manifest из сохранённых checkpoints; test inference включается явно после выбора решения.

## Проверки

```bash
python -m unittest discover -s tests -v
```

Тесты создают маленькие синтетические датасеты во временной директории и проверяют MLP/CNN, повторяемость seed, checkpoints, журнал, порядок предсказаний, экспорт submission и фабрику модели. Исходные данные и журнал реальных экспериментов не меняются.


## Подробное описание конфигов

См. [docs/configuration.md](docs/configuration.md): порядок слоёв, параметры, размеры тензоров и примеры scheduler.

`02_baseline_cnn.ipynb` остаётся для простого baseline. `04_augmentations.ipynb` содержит выбор одного из четырёх сравнимых CNN-рецептов на 256×256 и просмотр train-аугментаций. Порядок: `cnn_256_no_aug` → `cnn_256_geometry` → `cnn_256_photo` → `cnn_256_domain_shift`. У всех одинаковы архитектура и обучение, меняются только аугментации. CNN 96/128 сохранены в `configs/archive/`.

В обоих baseline-конфигах `image_size=256`: изображения подаются в исходном разрешении. `64` тоже допустимо, но означает уменьшение при загрузке и отдельный эксперимент. Исходные файлы никогда не меняются. MLP при 256×256 имеет около 50 млн весов только в первом Linear; уменьшение размера входа уменьшает её стоимость.

`training.scheduler` по умолчанию `null`. Поддерживаются StepLR (`step`), CosineAnnealingLR (`cosine`) и ReduceLROnPlateau (`plateau`). Scheduler обновляется после обучения и validation, задавая LR следующей эпохи. Plateau использует только `val_loss` либо `val_accuracy`. В истории сохраняются `learning_rate` текущей эпохи и `next_learning_rate`; LR печатается в логе и отображается на третьем графике. Состояние scheduler сохраняется в best/last checkpoints.


Поддержка встроенных ResNet и исправление deterministic для CUDA описаны в [docs/transfer_learning.md](docs/transfer_learning.md).

Новые обучения сохраняют вероятности в NPZ с IDs и порядком классов. В `05_error_analysis.ipynb` выбирается один EXPERIMENT_ID: ноутбук показывает ошибки с top-5 вероятностями и top-30 объектов с минимальным разрывом top-1/top-2 (TOP_K и UNCERTAIN_TOP_N настраиваются). Если val_probabilities.npz отсутствует, выполняется val-only inference best.pt на CPU/CUDA и результат сохраняется; переобучение не требуется. Сравнения TTA, ансамблей и аудит потенциала специалистов перенесены в `06_tta_ensembles.ipynb`. `src/inference.py` сохраняет сравнение и выбранный manifest отдельно от журнала обучений experiments.csv. См. [docs/inference.md](docs/inference.md).
