# Satellite Object Classification

Классификация спутниковых снимков по категориям объектов. Команда SBMTeam.

Финальный рецепт: AGMFA на ResNet50 и обычная ResNet50, сильные HardShift-аугментации, по 25 эпох со свежих ImageNet V2 весов на train+val. Для применения — Geometry-4 TTA и фиксированное смешивание вероятностей: поровну в общем случае, с повышенным весом AGMFA для зданий и ResNet50 для железной дороги.

```text
final/           Финальный код, ноутбук и конфиги для нового обучения
archive/         Исторические эксперименты: src, notebooks, configs, tests
docs/experiments.md  Короткая история гипотез и выводов
experiments.csv  Исторические обучения и будущие запуски
data/raw/        Исходные данные, скачиваются отдельно
artifacts/       Веса и логи новых финальных запусков
submissions/     Новые предсказания
```

В архиве сохранены ранние обучения, анализ валидации и их локальные артефакты. Исследования с размеченным тестом и связанные результаты удалены из рабочей версии. `final/` содержит рецепт для нового обучения; прежние веса и метрики не представлены как результаты этого рецепта.

Установка в окружении с Python 3.12+ и совместимыми PyTorch/torchvision:

```bash
python -m pip install -r final/requirements.txt
python -m final.src.runner train --raw-dir data/raw --run-id final_001
```

Обучение требует только `train.csv`, `val.csv`, `classes.csv`, `train/` и `val/`. Веса ImageNet скачиваются torchvision или берутся из его кэша. Для автономного запуска можно передать `--imagenet-weights /path/to/resnet50-11ad3fa6.pth`.

Применение после завершения обоих обучений:

```bash
python -m final.src.runner predict --agmfa-checkpoint artifacts/models/final_001_agmfa/last.pt --resnet50-checkpoint artifacts/models/final_001_resnet50/last.pt --sample-csv data/raw/sample_submission_example.csv --image-dir data/raw/test --output-dir submissions/final_001
```

Создаются CSV каждой модели и ансамбля, а также их вероятности. Из шаблона читаются только ID. Подробности — в [final/README.md](final/README.md), пошаговый запуск — в [финальном ноутбуке](final/notebooks/01_final_pipeline.ipynb).

Проверки на синтетических данных:

```bash
python -m unittest discover -s final/tests -v
python -m unittest discover -s archive/tests -t . -v
```
