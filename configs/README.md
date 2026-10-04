# Конфигурации экспериментов

Все рецепты — стартовые гипотезы. Результаты завершённых MLP находятся в experiments.csv и findings.md; MLP-конфиги оставлены для истории.

## Текущие CNN, вход 256×256

| Конфиг | Назначение | Эпохи |
| --- | --- | --- |
| baseline_cnn.json | Простой CNN без аугментаций для 02_baseline_cnn.ipynb | 40 |
| cnn_256_no_aug.json | Контроль для серии аугментаций | 45 |
| cnn_256_geometry.json | Геометрия | 45 |
| cnn_256_photo.json | Геометрия, цвет и gamma | 45 |
| cnn_256_domain_shift.json | Геометрия, цвет и искажения качества | 45 |
| cnn_256_balanced.json | Тот же рецепт, что geometry, сохранённое прежнее имя | 45 |
| cnn_256_deeper.json | Более ёмкая архитектура с геометрией | 60 |

Четыре рецепта no_aug/geometry/photo/domain_shift одинаковы по архитектуре и обучению. Сравнивайте их в 04_augmentations.ipynb. Начните с контроля, затем добавляйте преобразования.
Подробности диапазонов, вероятностей и архитектуры: [docs/augmentations.md](../docs/augmentations.md).
baseline_cnn имеет другую архитектуру и служит общим ориентиром, а не изолированным контролем аугментаций.
CNN 96/128 перенесены в archive/ с исходным содержимым; не входят в текущий план запусков.
Число эпох не гарантирует сходимости. Лучший checkpoint выбирается по val accuracy. Графики показывают переобучение/недообучение; фиксируйте изменения рецепта до следующего запуска.

## MLP и pretrained

MLP 64/96/128/256 оставлены без изменений. Законченные локальные запуски проверены: [MLP review](../docs/mlp_review.md).
Pretrained: resnet18_head_only, resnet18_partial, resnet18_full, resnet50_partial. Их конфиги не менялись; [режимы transfer learning](../docs/transfer_learning.md). Пока исследуем CNN.

## Запуск и воспроизводимость

```python
from src.experiment import run_experiment
result = run_experiment("configs/cnn_256_no_aug.json", exp_id="cnn_256_no_aug_s42_001")
```

Новый уникальный ID для каждого запуска; общий журнал требует последовательных запусков.
device=auto выбирает CUDA или CPU. pin_memory выключается на CPU; на Windows CPU num_workers устанавливается 0. Фактические настройки сохранены в config.json.
На Kaggle подключите GPU и при необходимости измените raw_dir на папку с train.csv/val.csv/classes.csv и train/val.
num_workers=2, persistent_workers=true, prefetch_factor=2 действуют при workers>0. Скорость Kaggle заранее не измерена; смотрите epoch_time_sec. Полчаса — ориентир, а не таймаут.

Seed фиксирован, deterministic=true и deterministic_warn_only=true. Совместимый pooling устраняет известную проблему CUDA AdaptiveAvgPool backward для кратных размеров. Warning mode не обещает побитовой повторяемости всех CUDA-операторов.
Test не читается без явного predict_test=True. Raw не меняется и статистики preprocessing на val/test не обучаются.
```json
"normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
```
Эта фиксированная нормализация CNN отображает RGB [0,1] в [-1,1]. ResNet использует ImageNet-нормализацию.

