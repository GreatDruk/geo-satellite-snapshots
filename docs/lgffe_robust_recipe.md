# Один robust-domain-shift recipe для LGFFE

Запуск: [13_lgffe_robust_domain_shift.ipynb](../notebooks/13_lgffe_robust_domain_shift.ipynb). Конфиг: [resnet50_lgffe_robust_augmix_mixstyle.json](../configs/resnet50_lgffe_robust_augmix_mixstyle.json). Новый ID: `resnet50_lgffe_robust_augmix_mixstyle_001`. Реальное обучение не запускалось. Подготовлен один конфиг, без абляций/сетки.

## Исходный результат

По сохранённым val probabilities и history обычная LGFFE stronger завершила 25 эпох, best epoch18, 668/699 (95.565%), 18.90мин. Stronger50 — 675/699. LGFFE исправила 6 его ошибок, добавила 13; 18 ошибок общие. Средняя confidence на ошибках LGFFE — 0.788, stronger50 — 0.893. Это согласуется с ручным наблюдением пользователя, но меньшая confidence сама по себе не означает улучшение Accuracy или calibration. Новые pretrained веса начинаются с ImageNet V2; прежний fitted LGFFE checkpoint для этого рецепта не требуется.

Дополнительные val данные при refit могли помочь из-за объёма, покрытия классов и/или домена. Один этот результат не доказывает недостаточность прежних аугментаций. Новый рецепт проверяет целиком предложенную комбинацию устойчивости; обещания прироста Accuracy нет.

## Pipeline

```text
RGB → resize полного кадра до 256, если требуется
    → один равновероятный D4 (8 вариантов, exact pixel transpose)
    → Robust AugMix
    → tensor / ImageNet normalization
    → stem → layer1 → MixStyle → layer2 → MixStyle → layer3 → layer4
    → LGFFE (GAP + GRU regional attention / weighted regional features)
    → concat / classification head → 20 logits
```

AugMix **заменяет** photometric/corruption часть stronger, а не дополняет её. Конфиг с одновременно указанными legacy аугментациями отклоняется. Crop, RRC, произвольные повороты, perspective, облака, erasing, CutMix/MixUp, CosFace и mining не добавлены.

### AugMix без JSD

Реализована смесь трёх независимых веток. Все стартуют с одного и того же полного кадра после D4. Каждая содержит равномерно случайные 1–3 операции, выбранные с возвращением из pool. Branch weights ~ Dirichlet(1,1,1), clean/augmented coefficient ~ Beta(1,1). Формула: `(1-m)*image + m*sum(w_i*branch_i)`. Смешивание выполняется до ImageNet normalization. Результат — RGB uint8 того же размера; clip/round сохраняют диапазон 0..255. Мягкие/почти исходные варианты — нормальная часть этой смеси, а не ошибочное отключение AugMix.

| Операция | Диапазон |
|---|---|
| Brightness / contrast | 0.35–1.6 / 0.4–1.6 |
| Gamma | 0.45–2.2; значения >1 затемняют |
| Saturation / hue | 0–1.8 / ±0.08 |
| Grayscale blend | доля grayscale 0.4–1 |
| White balance | независимые RGB gains 0.6–1.4 |
| Autocontrast / equalize | стандартные PIL операции |
| Solarize / invert | threshold64–192 / полная инверсия; редкий выбор |
| Gaussian blur | sigma0.4–2.5 |
| Motion blur | odd kernel3–13, direction0–180° |
| Defocus-like blur | нормированный disk PSF, radius1–4 |
| Resolution degradation | полный кадр downscale0.25–0.8 → upscale |
| JPEG | quality25–90 |
| Gaussian noise | std0.005–0.06 в шкале 0..1 |
| Poisson/shot noise | peak20–120; меньший peak даёт больше шума |

Относительный operation weight: 1 для обычных операций, 0.15 для solarize, 0.1 для invert. Это вероятности выбора операции, не веса смешивания веток. Motion/defocus — реальные свёртки line/disk PSF с отражёнными границами; geometry самого изображения не изменяется. Для motion допускается любое направление, без scipy и без установки новых библиотек.

Это custom robust operation pool с AugMix mixing, **не точное воспроизведение оригинального pool и JSD-протокола статьи**. Обучение использует один набор logits и обычный CE, без дополнительных forward passes.

### MixStyle

Два независимых parameter-free модуля после layer1 и layer2: `p=0.5`, `alpha=0.1`, `eps=1e-6`. При применении отдельно для каждого channel вычисляются spatial mean/std, статистики detach, normalized content сохраняется, а mean/std смешиваются с другим случайно переставленным объектом batch. Labels и spatial arrangement не перемешиваются. При batch1 модуль ничего не делает. Random permutation может оставить некоторые объекты на месте, как в стандартном алгоритме.

MixStyle зарегистрирован **вне frozen backbone subtrees**, чтобы существующий перевод layer1/layer2 в eval не выключал augmentation во время train/warmup. В `model.eval()` оба модуля возвращают исходный tensor без копирования и без RNG draws. У них нет parameters/state tensors; старые LGFFE checkpoints сохраняют прежние ключи и поведение. Новый checkpoint хранит включение MixStyle в config.

## Training / validation

ResNet50 IMAGENET1K_V2; partial layer3+layer4 и вся LGFFE/head; frozen backbone BN statistics как раньше. CE без label smoothing; 25 эпох, первые3 только новая голова. AdamW backboneLR5e-5/headLR5e-4, weight_decay0.001; cosineT_max25/eta_min1e-6; seed42; batch32. Все эти настройки совпадают с предыдущей LGFFE.

Val: RGB → resize if needed → tensor → ImageNet normalization. Нет AugMix, D4, MixStyle или TTA при выборе checkpoint. Test не читается; no test fitting. Общий pipeline автоматически сохраняет best.pt/last.pt, config/metadata/history/summary/training.log/learning_curves, val.csv/val_probabilities.npz и одну новую training-запись experiments.csv после завершения.

На Kaggle загрузить обновлённый `src`, новый JSON и notebook13; указать `RAW_DIR`, включить интернет для V2 weights либо подготовить torch cache; установить `RUN_TRAINING=True`. Synthetic preflight выполнится прежде training. При нехватке памяти уменьшить batch до первого запуска. Старые эксперименты не перезаписываются.

## Preview и проверки

В notebook сохранён выполненный preview train96.jpg (commercial_area), original +11 случайных вариантов, seed42. [Отдельное изображение](lgffe_robust_augmentation_preview.png). Можно выбирать другие **train** кадры через PREVIEW_CLASS/PREVIEW_INDEX; preview сохраняет Python/NumPy RNG состояния. Это визуальная диагностика transform, не оценка модели.

Команда проверки: `python -m unittest tests.test_robust_recipe tests.test_lgffe tests.test_transfer tests.test_augmentations -v`. 33 теста: 32 пройдены, один CUDA-only пропущен на CPU. Проверены настоящая torchvision ResNet50+LGFFE+MixStyle forward/backward, spatial attention normalization, warmup/partial, frozen BN buffers, LR groups; AugMix Dirichlet/Beta formula и branch depth; эффекты всех операций; exact D4; неизменность shape/range; seeded воспроизводимость; deterministic val без RNG; MixStyle train/eval и сохранение normalized spatial pattern; checkpoint restore/inference и полный synthetic training smoke во временной директории без test access. Прежние CNN/ResNet/LGFFE regression tests прошли. Реальные CUDA training и V2 download не проверялись локально.

Хеши 20 ранее существовавших файлов (journal, старые конфиги/notebook12, logs и predictions двух контрольных runs) совпали после подготовки и preview. Исходные данные и веса не записывались.

## Дополнительный frozen linear SVM в notebook13

После анализа neural run добавлен один SVM-блок. `RUN_TRAINING=False`, `RUN_SVM=True` и путь к `best.pt` текущего robust-LGFFE запуска. Локально этот checkpoint пока отсутствует: для фактического извлечения features нужен файл весов на Kaggle либо его локальная копия. Повторное neural training не требуется.

Извлекаются clean train/val **4096 concat global+local** перед всей classification head. Eval/inference_mode, все neural параметры frozen; AugMix/MixStyle/D4/TTA выключены. Train-only mean/std; fixed C=1; PyTorch MultiMarginLoss(p=1) с L2 весов и нерегуляризуемым bias, full-batch LBFGS до200 iterations, без validation tuning/early stopping. Objective: `0.5*||W||² + C*sum_i sum_{j!=y} max(0,1+s_j-s_y)`; нормировка на C*N меняет только численный масштаб. См. [MultiMarginLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiMarginLoss.html). Scores SVM не считаются вероятностями; остановка solver не объявляется доказанным точным optimum.

Feature cache: `artifacts/predictions/resnet50_lgffe_robust_augmix_mixstyle_001_frozen_features/`, с checkpoint/data/output hashes. Новый run: `resnet50_lgffe_robust_augmix_mixstyle_001_svm_001`. В models сохраняется `linear_svm.pt` с weight/bias, mean/std, class order и параметрами; в logs — parameters/summary, optimizer history, per-class errors и две confusion matrices; в predictions — val.csv, raw val_scores.npz и changed_objects.csv с id/true_label/softmax_pred/svm_pred. Сравниваются исходная softmax head и SVM на одинаковых val objects. Истинные val labels используются только после fitting для оценки. Основной neural checkpoint и experiments.csv не перезаписываются.

`python -m unittest tests.test_lgffe_svm -v`: четыре теста прошли. Проверены 20-классовое fitting/restore, чистое замороженное concat extraction, cache reuse, отдельные артефакты, train-only statistics, class order и подсчёт исправленных/новых ошибок. Реальные SVM результаты пока отсутствуют; test не используется.

## Источники

- [AugMix paper](https://arxiv.org/abs/1912.02781), [официальный mixing implementation](https://github.com/google-research/augmix/blob/master/augment_and_mix.py): stochastic chains, Dirichlet/Beta mixing.
- [MixStyle paper](https://arxiv.org/abs/2104.02008), [реализация авторов](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/modeling/ops/mixstyle.py): detached instance statistics, random batch permutation, train-only mixing.
