# Финальный рецепт

Обе модели обучаются заново: ResNet50 ImageNet V2, полный кадр 256×256, partial fine-tuning layer3/layer4, frozen BatchNorm, три эпохи прогрева головы. AGMFA использует исправленные conv2 attention features, три маски и multilayer fusion с GroupNorm.

Train: общий случайный D4 для clean/hard пары, один из шести обязательных HardShift-профилей, loss = 0.5 CE(clean) + 0.5 CE(hard) + 3 JS. Профили покрывают затемнение, снижение детализации и размытие, цветовой сдвиг, дымку, наклон съёмки и шум/JPEG. Бывшая val: только случайный D4 и CE с весом 2. Каждый исходный объект встречается один раз за эпоху. Train/val остаются в своих каталогах, поскольку их ID могут совпадать.

AdamW: backbone LR 5e-5, head LR 5e-4, weight decay 0.001. Для обеих моделей 25 эпох и cosine T_max=25. Сохраняется `last.pt` последней завершённой эпохи; для применения требуется завершённая эпоха 25. Метрики в логе относятся к обучению, включая бывшую val. Отдельной оценки, early stopping и выбора best checkpoint нет. Frozen BatchNorm и режим `model.eval()` при применении — обычные механизмы PyTorch.

Из корня репозитория:

```bash
python -m final.src.runner train --raw-dir data/raw --run-id final_001
```

Для одной модели добавьте `--model agmfa` или `--model resnet50`; затем вторую можно запустить с тем же run-id. Batch size задаёт число исходных объектов: clean/hard пара может увеличить число изображений до удвоенного batch size. Новый запуск требует нового ID. Точный resume прерванного обучения не реализован.

```bash
python -m final.src.runner predict --agmfa-checkpoint artifacts/models/final_001_agmfa/last.pt --resnet50-checkpoint artifacts/models/final_001_resnet50/last.pt --sample-csv data/raw/sample_submission_example.csv --image-dir data/raw/test --output-dir submissions/final_001
```

Geometry-4 — полный кадр и повороты 90°, 180°, 270°; усредняются softmax-вероятности четырёх просмотров. Базовый ансамбль — 50/50. Если оба top-2 базового ансамбля принадлежат группе `church/palace/commercial_area`, внутри группы доля AGMFA равна 0.75. Для `railway/railway_station` — 0.25. Суммарная вероятность группы и остальные классы остаются как в базовом ансамбле. Отбор группы идёт по предсказаниям.

Выходы: `agmfa.csv`, `resnet50.csv`, `ensemble.csv`, три NPZ с ID/классами/вероятностями и `recipe.json`. Новый каталог вывода защищает существующие результаты от перезаписи.
