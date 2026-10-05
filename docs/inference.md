# Вероятности, TTA и ансамбли

## Следующий Kaggle-запуск

В `03_pretrained.ipynb` уже выбран `configs/resnet101_partial_domain_shift_stronger.json` и новый ID `resnet101_partial_domain_shift_stronger_001`. Это копия сильного ResNet50-рецепта, меняется только архитектура: ImageNet V2, 256×256, partial layer3/layer4, warmup 3, 25 эпох, frozen BN, прежние optimizer/scheduler/аугментации. При нехватке GPU memory можно уменьшить batch с 32 до 16, сохраняя размер снимка.

Обучение сохраняет обычные best/last, логи, графики и строку `experiments.csv`. Дополнительно появляются `val_probabilities.npz` и, только при явном test inference, `test_probabilities.npz`. Для использования probability API не нужно переобучать старые checkpoints.

## Формат вероятностей

NPZ: `ids` — порядок изображений, `class_labels` — порядок столбцов, `probabilities` — [N, C], `true_labels` — только для размеченного split. Чтение без pickle. Проверяются уникальность IDs/классов, finite значения, диапазон [0,1], сумма строки и ground truth. Ансамбль выравнивает отдельно IDs и классы, отклоняет разные splits/классы/разметку. Нельзя просто усреднять столбцы разных файлов без выравнивания.

```python
from src.training.predict import load_probabilities, ensemble_probabilities
a = load_probabilities("artifacts/predictions/quality_inference_001/stronger__flips.npz")
b = load_probabilities("artifacts/predictions/quality_inference_001/mild__flips.npz")
ensemble = ensemble_probabilities([a, b])  # равные веса
scores = ensemble.table(include_scores=True)
```

Scored CSV содержит label, true_label, confidence, second_label, margin и entropy. Confidence — softmax, не измеренная калиброванная вероятность корректности. Логиты разных моделей напрямую не складываются.

## Варианты TTA

- `none`: исходное изображение.
- `flips`: original, horizontal, vertical, horizontal+vertical (4 уникальных вида).
- `d4`: четыре поворота кратно 90° с горизонтальным отражением/без него (8 уникальных видов, только квадратный вход).

Геометрия применяется к уже нормализованному тензору без интерполяции/обрезания. RGB mean/std постоянны в пространстве, поэтому отражение/поворот нормализованного изображения эквивалентны преобразованию перед нормализацией. Ни dropout, ни BN statistics не обновляются: eval + inference_mode, без optimizer/gradients. Усредняется softmax каждой ориентации. При совместном вычислении none/flips/d4 каждый уникальный вид проходит модель один раз, максимум 8 forwards на batch.

## Оценка на val

```python
from src.inference import run_validation_suite
result = run_validation_suite({
    "stronger": "artifacts/models/resnet50_partial_domain_shift_stronger_001/best.pt",
    "mild": "artifacts/models/resnet50_partial_domain_shift_001/best.pt",
    "resnet101": "artifacts/models/resnet101_partial_domain_shift_stronger_001/best.pt",
}, "quality_with_resnet101_001", device="auto", cache_from="quality_inference_001")
```

Включайте только существующие checkpoints. Аналогичные настройки есть в `06_tta_ensembles.ipynb`; по умолчанию он читает готовый локальный `quality_inference_001`, не повторяет inference. После Kaggle добавьте новый checkpoint, новый ID и включите `RUN_SUITE`.

`05_error_analysis.ipynb` посвящён одной модели: ошибки с top-k вероятностями и все val-снимки с наименьшим разрывом top-1/top-2. Функция validation_probabilities читает val_probabilities.npz, проверяет IDs/classes/true labels; при отсутствии файла восстанавливает best.pt, выполняет inference без TTA только на val и сохраняет NPZ. Случайные train-аугментации не применяются, test и experiments.csv не используются. Если существующий файл не соответствует val, функция выдаёт ошибку вместо его замены. Если в ноутбуке задан нестандартный PROBABILITIES_PATH, пользователь отвечает за выбор соответствующего модели файла: NPZ проверяет данные, но не содержит hash весов.

Suite создаёт отдельные `artifacts/logs/<ID>` и `artifacts/predictions/<ID>`: probabilities/scored CSV для каждого рецепта, inference.log, metadata.json с hashes/окружением/временем, comparison.csv и selection.json. Старые запуски не перезаписываются. Это inference-эксперимент, поэтому запись с вымышленным train_time в `experiments.csv` не добавляется; журнал оценок — comparison.csv.

`cache_from` позволяет использовать probabilities старых моделей без повторных forwards. Проверяются завершённость cache, hashes checkpoint/CSV/NPZ, IDs и ground truth; если полного cache для модели нет, её inference выполняется заново. В ноутбуке этот параметр уже указывает на локальный прогон. Перенесите его logs/predictions вместе с весами на Kaggle, чтобы использовать cache там.

Оцениваются все одиночные модели и равновесный ансамбль всех переданных моделей для каждого TTA-режима. При трёх и более моделях также сравниваются парные ансамбли: после ResNet101 это включает stronger+ResNet101, mild+ResNet101 и mild+stronger. Дополнительного inference для таких сочетаний нет — используются уже полученные вероятности. Никакого перебора весов. Manifest выбирается по Accuracy, затем NLL и числу views при совпадении метрик. Если ансамбль хуже одиночной модели, выбирается одиночная.

## Воспроизведение выбранного решения

`99_final_pipeline.ipynb` восстанавливает фиксированный selection.json. По умолчанию `PREDICT_TEST=False`; можно отдельно проверить replay на val. После окончательного выбора включите test inference: он не выбирает параметры, только применяет manifest, проверяет SHA-256 весов, сохраняет probabilities/predictions и submission в порядке sample_submission_example.csv.

```python
from src.inference import run_selected_inference
result = run_selected_inference(
    "artifacts/logs/quality_inference_001/selection.json",
    "final_val_replay_001", split="val", device="auto",
)
```

Относительные checkpoint paths в manifest переносимы вместе с репозиторием. Для inference weights download не выполняется. CUDA используется при наличии, иначе CPU; pin_memory включён только на CUDA, num_workers=0 по умолчанию безопасен на Windows. Test не читается validation suite; train не читается inference pipeline. Фиксированы seed 42 и deterministic warning mode; побитовая идентичность между различными CPU/CUDA версиями не обещается.

Специалист зданий остаётся условным следующим обучением после оценки ResNet101; сейчас не добавлен обязательный запуск трёх специалистов или full. Это соответствует бюджету `quality_next_steps.md`.
