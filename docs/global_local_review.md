# ResNet101 TTA и global/local: завершённая диагностика

2026-10-05. Run `resnet101_global_local_001`, CPU, batch4,4 потока,25.55 мин. Все699 val-изображений:8 ориентаций полного кадра +5 crops160 (углы/центр), resize→256 с сохранённой нормализацией. Исходный best101, seed42, eval/inference_mode; нового обучения и чтения test не было. CPU none повторяет все699 исходных Kaggle argmax.

## Работающие результаты

| Вариант | Правильно | Ошибок | Accuracy | NLL | Исправлено / испорчено против101 none |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 none + stronger50 D4,0.5/0.5 | 679 | 20 | 97.1388% | 0.132275 | 5 / 4 |
| 101 none | 678 | 21 | 96.9957% | 0.163261 | 0 / 0 |
| 0.75 global +0.25 local mean | 678 | 21 | 96.9957% | 0.135341 | 1 / 1 |
| 0.75 global +0.25 normalized local max | 677 | 22 | 96.8526% | 0.146651 | 0 / 1 |
| 101 flips + stronger50 flips,0.5/0.5 | 676 | 23 | 96.7096% | 0.128408 | 5 / 7 |
| 101 flips | 673 | 26 | 96.2804% | 0.146223 | 2 / 7 |
| 101 D4 | 673 | 26 | 96.2804% | 0.145953 | 3 / 8 |

Val Accuracy — основная метрика. Более низкий NLL не компенсирует потерянные правильные ответы. При равной Accuracy предпочитаем простой pipeline: mean-fusion не заменяет одиночную101 за меньший NLL и шесть views.

Повороты/отражения101 и простое crop-fusion не улучшили Accuracy. Сохраняем прежний смешанный ансамбль как лучший измеренный кандидат; одиночная101 остаётся более простой альтернативой всего с одной дополнительной ошибкой. Новый manifest выбран из работающих правил; отдельные crops исключены из выбора.

Полные таблицы: [final_comparison.csv](../artifacts/logs/resnet101_global_local_001/final_comparison.csv), [analysis_summary.json](../artifacts/logs/resnet101_global_local_001/analysis_summary.json), [final_selection.json](../artifacts/logs/resnet101_global_local_001/final_selection.json). Сохранены view NPZ/CSV, промежуточные chunks, per-class metrics и confusion pairs; оригинальные runs и experiments.csv не менялись.

## Что crops действительно показали

Из21 исходной ошибки11 становятся правильными хотя бы на одном crop:

- basketball_court2;
- church2;
- palace1;
- railway1, railway_station1;
- sea_ice1, terrace1;
- wetland2.

Для lake ни одна из4 ошибок не стала правильной на crops. Новое попадание true в top3 появилось только для1 ошибки: full уже имел true top3 на20/21.

Это **oracle potential, не Accuracy работающего решения**. На226 из678 исходно правильных снимков есть хотя бы один неправильный crop; на110 есть crop с неправильной confidence>=95%. Поэтому выбирать самый уверенный patch, голосовать или автоматически заменять глобальный ответ опасно. Качество отдельных crop-классификаторов всего81.83–87.84%.

Визуальные примеры: [global_local_examples.png](global_local_examples.png), все11 recovered cases показаны постранично в06.

- 1977: full уверенно tennis_court99.57%, bottom-left crop — правильный basketball_court99.90%. Более локальный вид действительно изменяет интерпретацию площадки.
- 3201: full terrace52% против true railway46.97%; центр/top-left становятся railway. Но другие crops уверенно предсказывают basketball_court: локальный максимум может принадлежать постороннему классу.
- 6146: full palace59.72%; bottom-left crop видит railway_station41.98%. Остальные локальные виды ошибаются, поэтому полезный сигнал не извлекается автоматически.

Гипотеза локальных признаков получает поддержку сразу в нескольких классах, но причиной может быть масштаб, удаление контекста или смена входного распределения. Диагностика не доказывает, что виноват исключительно average pooling. Сильный crop oracle и отсутствие прироста fixed fusion обосновывают **один обучаемый эксперимент**, а не агрессивную настройку весов на val.

## Следующий Kaggle-запуск

Конфиг: [resnet101_partial_avgmax_domain_shift.json](../configs/resnet101_partial_avgmax_domain_shift.json). Уже выбран в `03_pretrained.ipynb`, ID `resnet101_partial_avgmax_domain_shift_001`.

**Цель:** дать голове обучиться использовать сильные локальные активации вместе с усреднённым контекстом при одном backbone forward. Это более простой первый эксперимент, чем six-view MIL; не гарантированное решение crop-проблемы.

**Рецепт:** собственный best101 epoch21, spatial mean+max→4096 features, Dropout0.3,20 классов. Начальная голова `[W_old,0]`, bias_old сохраняется. На настоящей ResNet101 CPU-forward256 проверен: output(1,20), максимальная разница исходных/новых logits0.0 на проверочном входе. Все остальные backbone tensors переносятся строго.

- Input256, batch32, seed42.
- Partial layer3/layer4, frozen BN,1 head-warmup epoch, всего12 эпох.
- AdamW body1e-5/head1e-4, weight decay0.001; cosine12, eta_min1e-6.
- Имеющиеся stronger-аугментации сохраняются.
- Новые optimizer/scheduler; это fine-tuning весов, не восстановление старого optimizer-state.

На Kaggle нужен `artifacts/models/resnet101_partial_domain_shift_stronger_001/best.pt`; при другом расположении изменить training.initialize_from. ImageNet повторно не скачивается. Raw location при необходимости задать через data.raw_dir. Запустить03 сверху вниз; сохранить новые models/logs/predictions и новую строку журнала. Этот эксперимент **здесь не обучался**.

Сравнивать best нового run с сохранёнными101 none и текущим ансамблем, затем максимум одно равновесное объединение с исходной101. Не выбирать last вместо best. Исходная модель сохранена независимо от результата fine-tuning. Если нового одиночного/ансамблевого прироста нет, кандидат не включается в final pipeline.

## После этого

Специалист зданий — следующий резерв, если после avgmax сохраняется соответствующая путаница. Crops распознали лишь2/7 ошибок church и1/2 palace; lake не исправили. Не утверждаем, что один новый pooling решит все оставшиеся ошибки.

MIL, ResNet152/full и дополнительные более сильные corruptions сейчас не запускаем. `99_final_pipeline.ipynb` уже указывает на новый final_selection.json, пока с тем же лучшим ансамблем; PREDICT_TEST=False. Test — только inference окончательно выбранного решения.

Проверка реализации:49 тестов,48 passed и1 CUDA skip; тестируются геометрия, fixed fusion, class/ID alignment, resume, перенос весов/градиенты, полный synthetic fine-tuning с журналом и replay val/test. Синтетический test не относится к данным соревнования. Новый конфиг проверен настоящим CPU-forward; CUDA-training этого кандидата ещё не проверен.
