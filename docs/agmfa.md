# ResNet50 + AGMFA

Один новый кандидат: `configs/resnet50_agmfa_robust_augmix.json`,
`resnet50_agmfa_robust_augmix_fixed_001`. Запуск: `notebooks/14_resnet50_agmfa.ipynb`.
Это исправление текущей архитектуры, без изменения training recipe. Результатов
исправленного обучения пока нет; проверки на synthetic tensors не оценивают качество.

Источник: [Li et al., Attention-Guided Multilayer Feature Aggregation Network
for Remote Sensing Image Scene Classification, Remote Sensing 2021, 13(16), 3113](https://www.mdpi.com/2072-4292/13/16/3113),
разделы 3.1–3.3, уравнения 2–5.

## Реализованная схема

- В первом bottleneck `layer4` stride изменён на 1 у `conv2` и residual projection.
  Формы pretrained tensors не меняются; загружаются `IMAGENET1K_V2`.
- При 256×256: Res2 = 512×32×32, Res3 = 1024×16×16,
  Res4_1/2/3 = 512×16×16. Attention использует внутренние признаки каждого
  bottleneck после conv2 → bn2 → ReLU, до conv3. Затем вручную выполняются
  conv3 → bn3 → residual addition → final ReLU. Финальный Res4 для fusion
  остаётся полным выходом третьего блока: 2048×16×16.
- Три отдельных shared-MLP channel attention (avg + max, reduction16),
  затем spatial attention (channel mean/max → concat → Conv7×7 → sigmoid).
  MLP принимает 512 каналов. Shared MLP общий для avg/max внутри одной ветки;
  разные ветки независимы.
- Общая маска — произведение трёх spatial masks, диапазон [0,1].
  Spatial softmax, residual mask и дополнительный CBAM поверх fusion не добавляются.
- Res2 уменьшается через MaxPool2d(2); Res3/Res4 сохраняют 16×16.
  Общая маска умножается на эти исходные карты признаков.
  Concat допускает разные числа каналов: 512+1024+2048=3584.
  Общая Conv1×1 переводит их в 512 каналов, затем ReLU и GroupNorm(32,512).
- GAP → dropout0.3 → FC20, возвращаются raw logits. Transformer/QKV отсутствуют.

GroupNorm, fusion width512, reduction16 и dropout0.3 — явные настройки нашей
адаптации; опубликованные результаты статьи ими не воспроизводятся автоматически.
Сохранены ключевые элементы метода: высокое spatial resolution, несколько high-level
attention maps и guided multilayer fusion. Все новые параметры размещены под
`backbone.fc.*`, поэтому существующий transfer pipeline относит их к head LR group.
Стандартный ResNet avgpool/fc путь обходится при forward только у нового типа модели.

## Обучение и запуск

Текущий Robust AugMix + D4 скопирован целиком. MixStyle отсутствует.
Полный кадр 256×256, обычный CrossEntropyLoss без label smoothing и дополнительных
loss; seed42. AdamW: backbone LR5e-5, новые параметры LR5e-4, WD0.001.
Cosine T_max25/eta_min1e-6; 25 эпох. Первые 3 эпохи обучаются только новые
attention/fusion/head; затем также layer3/layer4. BatchNorm backbone всегда eval,
ранние параметры заморожены; GroupNorm fusion обучается с новыми параметрами.
Val использует только детерминированный full-frame resize и ImageNet normalization,
без AugMix, геометрии и TTA. Test не загружается.

В ноутбуке укажите `RAW_DIR`, затем `RUN_TRAINING = True` и выполните ячейки.
Это один запуск; перед ним автоматически выполняется synthetic preflight.
`RUN_PREFLIGHT = True` отдельно проверяет архитектуру без обучения на данных и
без скачивания pretrained weights. На GPU при нехватке памяти уменьшите
`BATCH_SIZE` с 32 до 16 до запуска; resolution остаётся 256. Прогноз времени
не измерен: layer4 на 16×16 требует больше памяти и вычислений, чем стандартные 8×8.

Общий pipeline сохраняет best.pt/last.pt в `artifacts/models/<exp_id>/`,
history.csv/summary.json/training.log/learning_curves.png в `artifacts/logs/<exp_id>/`,
val.csv/val_probabilities.npz в `artifacts/predictions/<exp_id>/` и одну запись
успешного запуска в experiments.csv. Исходные checkpoint не меняются.
`load_checkpoint` восстанавливает новый тип по сохранённому config без повторного
скачивания ImageNet weights; порядок классов и обычный inference сохраняются.

В исправленном config явно указано `model.agmfa.attention_source="conv2"`.
Для исторических checkpoint, где этого поля ещё нет, загрузчик сохраняет прежнюю
2048-канальную attention-схему (`block_output`), чтобы старые веса по-прежнему
загружались и выдавали прежние ответы. Это только совместимость истории;
новый запуск использует conv2. Старый config сохранён в
`configs/archive/resnet50_agmfa_robust_augmix_original.json`, старый исполненный
ноутбук с outputs — в `notebooks/archive/14_resnet50_agmfa_original.ipynb`.
Старые artifacts и experiments.csv не меняются.

## Проверки

`src/models/agmfa_checks.py` использует настоящую torchvision ResNet50, synthetic
batch 2×3×256×256 и два optimizer steps: warmup и partial. Проверяет shapes,
mask range/product, применение маски ко всем трём глубинам, конечные градиенты,
обновление новых и поздних параметров, неизменность ранних весов и BN buffers,
LR groups и повторяемый eval. Также каждый ручной bottleneck проверяется на
точное численное совпадение с обычным forward тех же весов; pre-hook conv3
проверяет совпадение извлечённых внутренних features. Attention при этой проверке
не применяется к backbone. Это техническая проверка; pretrained download
и реальные train/val/test для неё не нужны.

`tests/test_agmfa.py` дополнительно проверяет общий pipeline на маленькой residual
модели: переход после warmup, журнал, checkpoints, логи и графики, NPZ probabilities,
восстановление и совпадение inference. Реальные данные и журнал не затрагиваются.
