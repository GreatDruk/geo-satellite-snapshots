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
Исходные pretrained-конфиги resnet18_head_only, resnet18_partial, resnet18_full, resnet50_partial сохранены. Их результаты и выводы: [ResNet review](../docs/resnet_review.md); [режимы transfer learning](../docs/transfer_learning.md).

## Следующая серия ResNet50

| Конфиг | Что сравнивает |
| --- | --- |
| resnet50_partial_control.json | Уже проверенный partial-рецепт с отражениями; контроль, 94.85% у seed42 |
| resnet50_partial_domain_shift.json | Те же отражения + мягкие цвет/gamma/noise/blur/JPEG/downscale |
| resnet50_partial_domain_shift_stronger.json | Против мягкого варианта меняются только диапазоны blur и downscale |
| resnet50_full.json | Против partial-контроля меняется только режим разморозки; BN running statistics остаются frozen |

Все четыре: 256×256, batch32, seed42, ImageNet V2 weights и нормализация, 25epochs, head-only warmup3, AdamW (backbone LR=0.00005, head LR=0.0005, weight_decay=0.001), cosine T_max25/eta_min0.000001, head dropout0.3. Геометрия — horizontal/vertical flips по0.5, повороты не добавлялись.
У stronger blur=[0.5,2.5] вместо[0.2,1.0]px и down/upscale=[0.4,1.0] вместо[0.65,1.0]. Вероятности, цвет, noise/JPEG/gamma и max_corruptions=2 прежние. Вход всё равно256×256; кадр не обрезается, raw не меняется.

Порядок: мягкий domain_shift → stronger → при необходимости full. Контроль seed42 уже есть (`resnet50_partial_001`); новый control-файл — удобное имя того же рецепта, а не новый результат. Повтор нужен при смене окружения/протокола или для дополнительных seeds.

Запуск через существующий `03_pretrained.ipynb`: укажите путь к новому конфигу и **новый EXP_ID**. Например `CONFIG="configs/resnet50_partial_domain_shift.json"`, `EXP_ID="resnet50_partial_domain_shift_s42_001"`. Старый default ID в notebook уже использован, его нужно заменить. Pipeline сохранит графики, checkpoints и строку журнала.

После выбора перспективных вариантов повторите контроль и кандидата с seeds123/2026: `config["seed"] = 123` после `load_config`, отдельный EXP_ID на каждый seed. Не меняйте параллельно другие параметры. Фотоискажения применяются только к train, val остаётся исходной; диапазоны проверяются по train/val, не подбираются по test. Сильнее не обязательно лучше.

Full оставляет `freeze_bn=true` для сравнения с partial: running mean/variance заморожены, но разрешённые к обучению BN affine-параметры получают gradients. Предыдущий ResNet18 full имел `freeze_bn=false` и другие LR/epochs, поэтому это иной протокол.

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

