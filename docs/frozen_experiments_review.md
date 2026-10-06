# Frozen experiments: ансамбли и попарные классификаторы зданий

2026-10-06. Четыре фиксированных probability ensemble и три бинарных logistic classifier на frozen ResNet101 выполнены. Новая нейросеть не обучалась, test не читался. Full-контроль снят с плана по решению пользователя. Пользователь подтвердил разрешение финального train+val refit.

**08_frozen_pairwise_experiments.ipynb уже выполнен:** таблицы, routing и все изменённые изображения. Веса нейросетей, исходные predictions и experiments.csv не изменены. Comparison.csv в новых artifacts/logs служат журналом этой серии.

## Probability ensembles

Входы проверены по IDs, истинным меткам и порядку классов. Усредняются probabilities, не logits/голоса. Веса исходных моделей равны; D4 probabilities50 — один участник. Добавление к reference означает mean трёх исходных моделей, не50% новому участнику. Не делались calibration/сетки весов/выбор участников для отдельных известных ошибок.

| Рецепт | Правильно /699 | Accuracy | Fixes /breaks против reference | NLL |
|---|---:|---:|---:|---:|
| stronger50 single | 675 | 96.57% | 5 /9 | 0.1841 |
| ResNet101 single | 678 | 97.00% | 4 /5 | 0.1633 |
| **101 none + stronger50 D4: reference** | **679** | **97.14%** | — | 0.1323 |
| stronger50 + CosFace + mining | 676 | 96.71% | 3 /6 | 0.1496 |
| stronger50 + smoothing + CosFace + mining | 675 | 96.57% | 3 /7 | 0.1379 |
| 101 + stronger50 D4 + CosFace | 677 | 96.85% | 1 /3 | **0.1289** |
| 101 + stronger50 D4 + mining | 679 | 97.14% | 3 /3 | 0.1338 |

Против **stronger50**, тройка50/CosFace/mining исправляет4 и портит3: измеренный прирост ровно1 снимка. Полезность проверки diversity подтвердилась, но уже имеющийся pipeline679 лучше. Четвёрка со smoothing прироста Accuracy не дала.

Добавление mining к reference исправляет7790 wetland,10371 terrace,11562 palace; портит5900 railway_station,10032 church,10144 church. Тот же679, дополнительный forward и немного худший NLL: оставляем прежний reference.

Добавление CosFace исправляет10371 terrace; портит7548/10144 church и9099 commercial_area. Tennis4761 меняет один неверный ответ на другой. Улучшение NLL не выдаём за улучшение Accuracy.

Артефакты: `artifacts/logs/frozen_probability_ensembles_001/` (comparison, changes, per-class, metadata/source hashes) и `artifacts/predictions/frozen_probability_ensembles_001/` (NPZ/CSV всех фиксированных вариантов).

## Frozen embeddings и train-only logistic regression

Источник: исходный `resnet101_partial_domain_shift_stronger_001/best.pt`, SHA256 `753850b365640e0ca2c6d4df49b722d5e9ce846d3eef11367962ab306107775b`. Получены2048 признаков перед fc/dropout с чистого полного кадра, исходные deterministic resize/нормализация. Model.eval, inference_mode, без TTA, аугментаций, обновления BN и градиентов backbone.

Из train нужны только749 объектов трёх классов (church250, commercial250, palace249); извлечены все699 val для проверки регрессий. CPU inference занял453.7сек. Исходная голова воспроизвела все699 Kaggle argmax. Кэш содержит IDs/labels/features/probabilities/class order и hashes checkpoint/CSV/NPZ; повторный запуск использует завершённый кэш.

Для каждой пары стандартизация fit только на её train-признаках. C=1 заранее фиксирован; objective=sum(binary CE)+||w||²/(2C), bias не штрафуется. Бинарная L2 logistic regression решается имеющимся torch LBFGS, без установки sklearn/scipy. Коэффициенты, mean/std и convergence сохранены в JSON без pickle. [Определение LogisticRegression/L2](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [torch LBFGS](https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html).

| Пара | Train samples | Train Accuracy | Val истинных классов пары | Итераций |
|---|---:|---:|---:|---:|
| church /palace | 499 | 100% | 66/70 =94.29% | 37 |
| church /commercial_area | 500 | 100% | 67/70 =95.71% | 35 |
| palace /commercial_area | 499 | 100% | 66/70 =94.29% | 34 |

Все solver converged, max gradient≈6.8e-7–8.6e-7. Fit трёх classifiers вместе с сохранением/анализом≈10сек. Val/test не участвовали в scaling/fitting/выборе C/порогов. Train100% не является независимой проверкой обобщения.

## Top-2 reranking

Исходные top2 определяют неупорядоченную пару. Если она соответствует classifier, выбираем между двумя классами по binary logit>=0 независимо от confidence. Все остальные labels сохраняются. True label используется только после inference для отчёта. Одни classifiers применены к двум заранее заданным основаниям: исходной101 и прежнему лучшему ансамблю.

| Основание | До | После | Fixes /breaks | Routed | Changed |
|---|---:|---:|---:|---:|---:|
| ResNet101 single | 678/699 | 677/699 | 1 /2 | 83 | 3 |
| 101 + stronger50 D4 | 679/699 | 677/699 | 1 /3 | 87 | 4 |

**Top2 не защищает все правильные объекты.** У ансамбля routed87, из них79 исходно правильных. В этой серии истинный класс всегда принадлежал выбранной паре, но даже при этом баланс отрицательный; для других данных принадлежность не гарантирована.

Church/palace не поменял ни одного из39/42 routed решений. Church/commercial исправляет12684 church→commercial, но ломает7548 church у ансамбля или9099 commercial у одиночной101. Palace/commercial добавляет регрессии без исправлений.

Все изменения относительно ансамбля:

| ID | Истинный класс | Было | Стало | Результат |
|---|---|---|---|---|
| 12684.jpg | church | commercial_area | church | fix |
| 2200.jpg | palace | palace | commercial_area | break |
| 7548.jpg | church | church | commercial_area | break |
| 9099.jpg | commercial_area | commercial_area | palace | break |

[Галерея изменений](frozen_experiments_figure_03.png). Palace2200 — комплекс/парк с водоёмами, church7548 — крыша архитектурного комплекса, commercial9099 — квартал зданий. Выбранная граница помогает одному church12684, но этого недостаточно для общего выигрыша. Экспертов не применяем выборочно к заранее известным неправильным IDs, labels не меняем.

Binary probability условна внутри пары, не является20-классовой probability. Поэтому сохраняются labels/routing, без искусственной NLL всей системы.

Артефакты: `artifacts/predictions/building_pairwise_features_001/`; `artifacts/models/frozen_building_reranker_001/pair_*.json`; `artifacts/logs/frozen_building_reranker_001/` (binary/routing/changes/per-class/comparison); `artifacts/predictions/frozen_building_reranker_001/` (готовые labels).

## Решение и финальный этап

1. Оставляем **101 none + stronger50 D4,679/699**, равные веса. Дополнительные CosFace/mining и pairwise не включаем. Если нужен один forward, исходная101 даёт678; это выбор стоимости inference, не выигрыш новых методов.
2. Full-контроль больше не предлагается; новых сеток/семян нет. Эти эксперименты не доказывают абсолютный потолок ResNet или нерешаемость общих ошибок: проверены конкретные фиксированные методы на имеющихся признаках.
3. По подтверждённым пользователем правилам разрешён **final refit на train+val5698**. Это единственный ещё не выполненный тяжёлый этап: даст размеченные реальные target-domain примеры, но размер прироста заранее неизвестен. Честной Accuracy на прежней val после этого нет.
4. Подготовлены `resnet50_final_train_val.json` и `resnet101_final_train_val.json`: partial+stronger, прежние optimizer/LR/aug/frozenBN/warmup/seed42. Fixed epochs по исходному best:50=20,101=21; исходный cosine T_max25 сохранён. Повтор выбранного рецепта из ImageNetV2 на объединённых данных, не новое fine-tuning старого best с новым LR. Для двухмодельного финала нужны оба refit; для single101 только101. Проигравшие модели/reranker автоматически не переобучаем.
5. **src/final_refit.py отдельно от обычного training**, notebook09 с RUN_REFIT=False. Сохраняет fixed last **final.pt**, config/metadata/history/log/train curves; experiments.csv получает строку с пустой val_accuracy. Test predictions отдельно включаются. Никакого early stopping/выбора checkpoint на бывшей val. Реальные нейросети сейчас не переобучались, нужен Kaggle GPU. После refit финальный inference должен явно использовать новые final.pt: прежний99/manifest сам на них не переключается.

Оба новых ноутбука выполнены (09 без обучения). Все27 тестов pairwise, final_refit и существующих inference/pipeline прошли. Синтетический refit проверяет оба split, последнюю эпоху, пустую val_accuracy, отсутствие неявного test access, неизменность raw и совместимость final.pt с load_checkpoint. Обычная логика train/val сохранена; добавлены только проверки против запуска final-конфига через run_experiment и оценки final-refit checkpoint как held-out val через run_selected_inference. Зависимости не устанавливались.
