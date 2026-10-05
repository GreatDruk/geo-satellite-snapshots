# Реальная оценка TTA и ансамбля ResNet50

2026-10-05. Восстановлены два best checkpoints; CPU, float32, eval/inference_mode, все 699 val-снимков. Ни обучения, ни чтения test/train. Никаких новых pretrained downloads. Время всего CPU-прогона: 28.44 мин; это inference, не GPU train time.

| Рецепт | Accuracy | Ошибок | NLL | Views/image |
| --- | --- | --- | --- | --- |
| ensemble__flips | 96.85% | 22 | 0.1545 | 8 |
| ensemble__d4 | 96.71% | 23 | 0.1537 | 16 |
| ensemble__none | 96.71% | 23 | 0.1686 | 2 |
| mild__d4 | 96.57% | 24 | 0.1726 | 8 |
| stronger__none | 96.57% | 24 | 0.1841 | 1 |
| stronger__flips | 96.42% | 25 | 0.1672 | 4 |
| mild__flips | 96.28% | 26 | 0.1833 | 4 |
| mild__none | 96.14% | 27 | 0.2081 | 1 |
| stronger__d4 | 95.99% | 28 | 0.1675 | 8 |

Выбран ensemble__flips: 677/699, 22 ошибок. Против исходного stronger исправлено 5, добавлено 3; чистый прирост 0.29 п.п. Все рецепты сравнивались по одной val; результаты не обещают такой же test score. Критерий выбора: Accuracy, затем NLL и число views.

TTA для stronger отдельно проиграл: flips 25 ошибок, D4 28 вместо 24. Ниже NLL не означает выше Accuracy. Лучшая комбинация определяется измерением, TTA не включается принудительно.

Общее качество выросло, но ошибки church увеличились с 7 до 9; lake остаётся с 4 ошибками, basketball_court — с 3. Поэтому ансамбль не решает целевую путаницу зданий и не объявляется лучше по каждому классу. Следующий ResNet101 нужен прежде всего для более различительных признаков и нового взаимодополнения; специалист зданий остаётся резервом после его оценки.

Исходные argmax обоих checkpoints совпали с сохранёнными CSV для всех 699 снимков. Проверены hashes checkpoint/val.csv/classes.csv, а также неизменность experiments.csv. В metadata.json добавлены hashes NPZ для безопасного cache. Результаты: comparison.csv, probabilities/scored CSV, selection.json, selected_class_metrics.csv и selected_confusions.csv в inference-артефактах. Новые обучения автоматически сохраняют probabilities.

В 06_tta_ensembles.ipynb сохранены таблица результатов, per-class metrics, confidence/margin, галерея ошибок выбранного рецепта и список регрессий против stronger. 99_final_pipeline.ipynb применяет manifest; test по умолчанию выключен.

Следующий обязательный новый запуск: configs/resnet101_partial_domain_shift_stronger.json, выбранный в 03_pretrained.ipynb. После него добавить best.pt в CHECKPOINTS и новый inference ID: существующие probabilities переиспользуются через cache_from. Будут оценены одиночные, парные и общий равновесные ансамбли. Специалист зданий остаётся резервным вторым обучением по результатам ResNet101, а не обязательным запуском сейчас.
