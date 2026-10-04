# Pretrained ResNet и режимы обучения

Встроена поддержка torchvision ResNet18/34/50/101/152. Стартовые конфиги подготовлены для ResNet18 и ResNet50; AlexNet, VGG и Inception не реализованы встроенным адаптером и не получили конфигов. Они остаются разрешёнными архитектурами для будущей внешней фабрики. DenseNet и Transformers не используются.

## Готовые конфиги

| Конфиг | Режим | Эпохи | LR тела / головы | Размораживаемые блоки |
| --- | --- | --- | --- | --- |
| resnet18_head_only.json | head_only | 15 | — / 0.001 | Только новая fc |
| resnet18_partial.json | partial | 25 | 0.0001 / 0.0005 | layer4 + fc |
| resnet18_full.json | full | 30 | 0.00003 / 0.0003 | Вся сеть |
| resnet50_partial.json | partial | 25 | 0.00005 / 0.0005 | layer3, layer4 + fc |

Это предложенные настройки без результатов экспериментов. Начните с resnet18_head_only, затем resnet18_partial/full. ResNet50 — более ёмкая альтернатива, не доказанное улучшение на этих данных.

Во всех конфигурациях вход 256×256, batch=32, AdamW с weight_decay=0.001, cosine scheduler, seed=42 и device=auto (CUDA при наличии, иначе CPU). Используется ImageNet-нормализация mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]. Полный кадр сохраняется: код делает resize при необходимости и не использует центральный crop из стандартного torchvision-предобработчика. Это выбранный протокол для спутниковых изображений, его качество нужно проверить на val.

## Model

```json
"model": {
  "name": "resnet18",
  "pretrained": true,
  "weights": "IMAGENET1K_V1",
  "mode": "partial",
  "unfreeze_blocks": ["layer4"],
  "head_dropout": 0.3,
  "freeze_bn": true,
  "warmup_epochs": 3
}
```

- `name`: конкретная архитектура. Также принимается `name=resnet` с `architecture=resnet18/resnet34/...`.
- `pretrained=true`: загрузить официальные ImageNet-веса. `false` строит ту же архитектуру с нуля; заморозка случайных признаков обычно нецелесообразна.
- `weights`: имя набора весов torchvision. ResNet18-конфиги используют IMAGENET1K_V1; ResNet50 — IMAGENET1K_V2. DEFAULT поддерживается, но фактическое имя сохраняется в конфиге запуска, чтобы оно не менялось незаметно между версиями.
- `mode=head_only`: все слои до fc заморожены. Исходная fc заменяется Dropout → Linear на число классов из classes.csv. Обучаются только эти новые веса.
- `mode=partial`: кроме новой головы обучаются перечисленные в unfreeze_blocks блоки. Возможные имена: conv1, bn1, layer1, layer2, layer3, layer4. Ошибочное имя отклоняется.
- `mode=full`: все параметры могут обучаться. unfreeze_blocks для этого режима не задаётся.
- `head_dropout`: dropout перед новой последней Linear. Модель возвращает logits без Softmax.
- `freeze_bn=true`: сохранять running mean/variance всех BatchNorm в eval mode даже в размороженных блоках. Их affine-параметры остаются обучаемыми, если сам блок разморожен. `false` разрешает обновление статистик в размороженных блоках. В полностью замороженных блоках статистики всегда фиксируются независимо от этой настройки.
- `warmup_epochs=3`: первые три эпохи обучать только голову, начиная с четвёртой включить заданный partial/full. В head_only это поле не нужно. Укажите меньше общего числа эпох. Scheduler продолжает общий график, не перезапускается при разморозке.

Только requires_grad=false недостаточно для настоящей заморозки: BatchNorm может менять буферы, а Dropout — поведение. Обёртка TransferClassifier после каждого model.train() возвращает замороженные поддеревья в eval mode. Это проверено тестами, включая неизменность буферов BatchNorm.

## Разные скорости обучения

```json
"optimizer": {
  "name": "adamw",
  "lr": 0.0005,
  "backbone_lr": 0.0001,
  "head_lr": 0.0005,
  "weight_decay": 0.001
}
```

`lr` используется по умолчанию для группы, если её отдельный LR не указан. Тело и голова получают отдельные param groups; полностью замороженные блоки в optimizer не включаются. Во время warmup параметры будущих размороженных блоков уже присутствуют в группах, но не получают gradients и не обновляются. Это сохраняет состояние optimizer и scheduler при переключении режима.

В history.csv сохраняются training_mode, learning_rates и next_learning_rates — JSON-объекты LR всех групп. Старые числовые learning_rate/next_learning_rate содержат LR первой группы для совместимости. График показывает отдельно голову и тело. Конфиг, режим, веса, optimizer и scheduler сохраняются обычным pipeline; experiments.csv получает архитектуру и pretrained=true.

## Deterministic на CUDA

Наиболее вероятная причина прежнего сбоя cnn_256 — backward AdaptiveAvgPool2d при строгом deterministic. Без traceback конкретный сбой Kaggle не подтверждён. PyTorch прямо описывает ограничение: [torch.use_deterministic_algorithms](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).

CompatibleAdaptiveAvgPool2d использует обычный AvgPool2d, если входные размеры делятся на выходные; для глобального pooling 1×1 — mean по пространственным осям. Это эквивалентные операции для таких размеров, проверены значения и gradients. Они покрывают pooling наших CNN 256 и глобальный pooling ResNet. Для некратных размеров сохраняется исходный adaptive pooling, чтобы не менять определение перекрывающихся окон.

В CNN/pretrained-конфиги добавлено:

```json
"deterministic": true,
"deterministic_warn_only": true
```

Seed и cuDNN-настройки сохраняются. Если другой оператор не имеет deterministic-реализации, PyTorch предупреждает вместо RuntimeError. Такой режим **не гарантирует побитовую повторяемость** для этих операций. Для строгой проверки используйте deterministic_warn_only=false; unsupported операции снова будут отклоняться. Молчаливого переключения режима после ошибки нет. Политика сохраняется в config.json/metadata.json.

## Запуск и зависимости

```python
from src.experiment import run_experiment

result = run_experiment("configs/resnet18_partial.json", exp_id="resnet18_partial_s42_001")
```

Используется тот же pipeline, отдельная фабрика не нужна. Checkpoint восстанавливается обычным load_checkpoint: ImageNet-веса повторно не скачиваются, состояние берётся из checkpoint. Для точного продолжения прерванного обучения по-прежнему нужен отдельный механизм сохранения RNG/загрузчиков; текущий load_checkpoint восстанавливает модель для оценки/inference.

Нужен torchvision, совместимый с установленным torch. Зависимость добавлена в pyproject.toml, локально пакет не устанавливался. На Kaggle проверьте, что импорт torchvision работает. Для первого получения ImageNet-весов нужен доступ к сети либо уже заполненный torch hub cache; код не заменяет отсутствующие pretrained-веса случайными молча. Скачивание выполняется только при реальном запуске pretrained-конфига, не при импорте src.

Локальные тесты проверяют режимы на маленьком тестовом backbone, pooling, группы LR, warmup, журнал и восстановление без скачивания. Отдельные реальные torchvision/CUDA тесты запускаются только при наличии этих зависимостей; в текущем локальном окружении они пропущены. Ни GPU-скорость, ни качество ImageNet-весов на соревновании пока не измерены.

Официальные источники: [ResNet](https://docs.pytorch.org/vision/stable/models/resnet.html), [Transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

На CPU pin_memory автоматически выключается; на Windows также устанавливается num_workers=0. Эти фактические настройки сохраняются в конфиге запуска. При явном device=cuda отсутствие CUDA по-прежнему считается ошибкой; для переносимого запуска используйте auto.
