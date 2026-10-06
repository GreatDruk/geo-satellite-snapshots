# ResNet50: итог проверки crop, smoothing, CosFace и mining

Сравнение одиночных моделей на исходной val699 с `resnet50_partial_domain_shift_stronger_001`. Без TTA/ансамблей. Найдены новые completed runs smoothing, CosFace, hard_mining; прежний RRC включён как контекст. В журнале/папках не найдены completed `rrc_smoothing` и `rrc_cutmix`, поэтому их эффективность не оценивалась.

Полные числа/hashes: [audit JSON](resnet50_strategies_audit.json). Таблицы: [comparison](resnet50_strategies_comparison.csv), [per-class](resnet50_strategies_classes.csv), [fixes/breaks по IDs](resnet50_strategies_changes.csv), [confusions](resnet50_strategies_confusions.csv), [confidence](resnet50_strategies_confidence.csv).

## Результат: stronger остаётся лучшим по Accuracy

| Рецепт | Accuracy | Правильно | Ошибки | Исправлено / испорчено против stronger | NLL | Best epoch | Обучение, мин |
|---|---:|---:|---:|---:|---:|---:|---:|
| stronger50 | **96.57%** | **675** | **24** | — | 0.1841 | 20 | 18.85 |
| RRC | 95.14% | 665 | 34 | 4 / 14 | 0.2212 | 24 | 19.20 |
| smoothing0.05 безcrop | 96.14% | 672 | 27 | 6 / 9 | 0.2090 | 19 | 20.91 |
| CE + auxiliary CosFace | 96.42% | 674 | 25 | 5 / 6 | 0.1907 | 22 | 18.27 |
| hard mining5% | 96.14% | 672 | 27 | 8 / 11 | **0.1773** | 22 | 19.90 |

Ни один кандидат не дал положительного баланса ответов. CosFace практически на уровне контроля, разница — один снимок; это не доказательство принципиальной слабости метода. Mining немного лучше по NLL, но соревнование оценивает argmax Accuracy. Повторов seeds/сетки весов не требуется, чтобы оставить measured winner675/699. Улучшение NLL/снижение confidence не выдаётся за улучшение Accuracy.

## Аудит выполнения

Все25 эпох finite и complete; configs соответствуют подготовленным JSON, warmup1–3/head_only и4–25/partial правильные, LR-группы соответствуют cosine25. Journal/summary/history/CSV/NPZ согласуются. IDs/classes/ground truth выровнены на699 снимков. Val-loss — обычный unsmoothed CE, не smoothing/CosFace train-loss. Нет признаков случайно включённых crop/CutMix или другого режима backbone в этих трёх запусках.

Проверены новые best.pt: epochs19/22/22, configs/score/class order совпадают, state tensors finite. У CosFace auxiliary_head присутствует и её параметры входят в head optimizer group. То есть её не забыли добавить в optimizer. Hashes весов сохранены в audit; actual gradients всего Kaggle-run задним числом не восстанавливаются. Smoothing/mining checkpoints проверены структурно, их полный inference повторно не выполнялся.

## Что именно дали методы

### Smoothing: меньше уверенности, но не лучше классификация

Средняя confidence всей val падает0.980→0.896, confidence ошибочных ответов0.893→0.661. Ошибок с confidence≥0.95:12→0. Значит smoothing действительно повлиял на поведение; это не «параметр проигнорировался».

Но accuracy672/699 ниже675, NLL0.2090 выше0.1841. Confidence правильных ответов тоже снизилась. ECE с10 фиксированными bins на этой val:0.0278→0.0657, то есть даже по этому приближённому критерию калибровка не улучшилась; модель стала в среднем недоуверенной. Это оценка одного набора, не универсальная характеристика calibration.

Исправлены lake1617, overpass1685, terrace5108, tennis8132, church10032, station13878. При этом добавлены palace32/11490/11562, church7548/10472, commercial6989 и др. Ошибки palace0→3, commercial2→3, church7→8. Train-loss≈0.356 нельзя напрямую сравнивать с CE≈0.007: у smoothing иная целевая функция с ненулевым теоретическим минимумом; train Accuracy99.88% подтверждает отсутствие очевидного недообучения.

### CosFace: не решил главную семантическую группу

Lake4→3, tennis2→1, station2→1; church7→8, palace0→1, wetland0→1, railway0→1. В частности church→palace4→5 и church→commercial2→3. Ключевая гипотеза о более полезном разделении зданий выбранным auxiliary margin не подтвердилась.

Важно: это **основной CE classifier с auxiliary CosFace objective**, не чистый CosFace вместо всей inference-головы. Для проверки возможного скрытого сигнала выполнен один отдельный CPU-forward val через уже обученный backbone: сравнили обычную голову и label-free cosine голову, scale16, **без true-class margin**. Основная голова воспроизвела699/699 Kaggle argmax, max probability difference8.35e-6. Auxiliary alone дал669/699=95.71%,30 ошибок, NLL0.1782;6 исправлений и12 регрессий против stronger. Переключать финальный classifier на auxiliary невыгодно. Усреднение голов и подбор scale/весов не выполнялись.

Диагностические probabilities/CSV: `artifacts/predictions/resnet50_cosface_head_check_001/`; summary сhash источника: `artifacts/logs/resnet50_cosface_head_check_001/summary.json`. Это inference diagnostic, не новое обучение, строка вexperiments.csv не добавлена.

### Mining: сигнал есть, нет выигрышного общего баланса

Восемь исправлений — больше всех новых кандидатов. Commercial2→1, basketball3→2, tennis2→1, overpass1→0. Но church7→9, palace0→2, railway0→2, rectangular_farmland0→1. Исправил basketball6931 и обе исходные commercial ошибки45/9099, но добавил другие semantic confusions. Хороший NLL0.1773 не компенсирует11 регрессий.

Проверены все25 mining CSV и history. Каждая эпоха сохраняет4999 уникальных оригинальных IDs; после5-й эпохи добавляются ровно249 повторов, всего5248 train views. IDs сдвинуты на следующую эпоху правильно; observations повторяемых samples равны2. За20 активных эпох добавлено4980 views, затронуто1773 уникальных train-примера. Это не ошибка sampler, не потеря train-примеров и не sampling по val.

К epoch23–25 лишь3–5 train-примеров имеют отрицательный margin, а выбираются249. Их средний margin≈0.90: поздние «hard» — преимущественно уверенно правильные ответы. Поэтому фиксированное повторение5% уже плохо соответствует цели атаковать реальные val-ошибки. Самые повторяемые IDs: farmland1578/10142 по18 раз, terrace5834 —17, church12876 —16. По группам чаще повторялись palace654 иchurch590 views, но val зданий не улучшилась. Difficulty measured on случайной train-аугментации сdropout; её нельзя приравнять к трудности чистого изображения или target-domain val.

## Где остаётся проблема

| Класс | stronger | RRC | smoothing | CosFace | mining |
|---|---:|---:|---:|---:|---:|
| church | 7 | 9 | 8 | 8 | 9 |
| lake | 4 | 6 | 3 | 3 | 4 |
| basketball_court | 3 | 3 | 3 | 3 | 2 |
| tennis_court | 2 | 1 | 1 | 1 | 1 |
| commercial_area | 2 | 2 | 3 | 2 | 1 |
| palace | 0 | 3 | 3 | 1 | 2 |
| railway | 0 | 1 | 1 | 1 | 2 |

У stronger+smoothing+CosFace+mining **15 общих ошибочных объектов**:6church,3lake,2basketball,1tennis,1baseball,1station,1terrace. Это не обязательно одна и та же неправильная метка: например tennis4761 у stronger/mining отвечаетchurch, уsmoothing/CosFace —medium_residential. Часто предсказанный неверный класс всё же совпадает между всеми моделями.

Просмотрены все15: [галерея с четырьмя предсказаниями](resnet50_strategies_common_errors.png). Сильно размытый basketball1977, почти лишённый деталейcourt10003, пикселизованныйbaseball12286 — проблемы качества/доступного сигнала. Но church10647/10144/12850 показывают более чёткие архитектурные комплексы: их путаницу нельзя объяснить одним blur. Lake13048/13473 содержат смесь воды/растительности, lake12301 — горный водоём среди снежных/светлых склонов, конкурирующий соsnowberg. Наблюдения не доказывают ошибки разметки; изображения/истинные метки не изменялись.

Все три новых метода одновременно испортили два ранее правильных ответа: palace32→commercial иchurch10472→palace. [Галерея общих регрессий](resnet50_strategies_common_regressions.png). Это иллюстрирует смещение границ соседних классов: метод может исправлять одни пограничные примеры и ухудшать другие.

## Кривые и выводы

[Кривые](resnet50_strategies_curves.png), [все классы](resnet50_strategies_classes.png). Все модели достигают≈99.7–99.9% train Accuracy. Train-loss различается по определению, train mining Accuracy также включает повторы. Валидировать capacity или сравнивать regularization по одному train loss неправильно.

Средняя Accuracy последних7 эпох: stronger96.20%, smoothing95.85%, CosFace95.93%, mining95.95%. Следовательно, выбор единственной эпохи не объясняет весь результат. У CosFace/mining минимум val NLL около13-й эпохи, а max Accuracy22-й; ранний NLL и лучший argmax — разные критерии. Увеличение эпох, увеличение smoothing или ещё больше повторов не имеют сейчас убедительного основания.

**Итог:** предыдущие идеи были разумными экспериментальными гипотезами, но ни одна не улучшила заданную метрику. Smoothing изменил confidence; mining изменил распределение ошибок/снизил NLL; CosFace оставил≈тот же уровень. Сильная модель может почти идеально запомнитьtrain и при этом неверно обобщать на изменённых сценах. Признаки/данные важнее дальнейшего подавления confidence. Отрицательный итог серии следует признать, а не объявлять улучшением по вспомогательной метрике.

## Следующий шаг: один контроль адаптации backbone

1. Сохраняем stronger50 как победителя675/699. Новые recipes остаются для истории. Не комбинируем проигравшие loss/mining/crop и не предлагаем новый ансамбль.
2. Если продолжаем одно новое обучение на50, наиболее ясный ещё не выполненный контроль — **full fine-tuning с теми же сильными аугментациями**, без новыхloss/crops/mining. Прежний `resnet50_full_001` имел только flips и93.71%; это полезный отрицательный результатfull на том режиме, но не измерениеfull+stronger. Не выдаём его за уже проведённое сравнение.
3. Гипотеза: позволить адаптироваться conv1/layer1/layer2, которые сейчас остаются ImageNet-fixed. Это тест ограничения текущей feature adaptation, а не увеличение ширины/глубины сети. Early features могут быть достаточно универсальны иfull может усилить overfitting; поэтому это один контроль, не обещание прироста.
4. Для изолированного сравнения: ImageNetV2,256,batch32,25epochs,seed42,warmup3,frozenBN,AdamW body5e-5/head5e-4,cosine25 — прежние. Меняется только mode partial→full и убираетсяunfreeze_blocks. Никакого одновременного изменения LR/aug/loss. Время может быть выше; прошлыйfull занял31.27мин, новый ещё не измерен. Проверять Accuracy иfixes/breaks противstronger50, отдельноchurch/palace/lake; отрицательный баланс означает оставитьpartial.
5. До финального refit выяснить разрешениеtrain+val по **правилам соревнования**, не по чужому ответу. Если явно разрешено, после выбора рецепта обучение на всех допустимых размеченных данных даст доступ к реальным target-domain примерам. Эпохи фиксируются заранее; бывшаяval не служит независимой оценкой или источником выбораcheckpoint. Test остаётся inference-only. Это отдельный финальный этап, не способ дорисовать более высокую val Accuracy.

Для пункта4 подготовлен `configs/resnet50_full_domain_shift_stronger.json`, default ID `resnet50_full_domain_shift_stronger_001`. Старые full/partial конфиги не изменены.03 выбирает новый контролируемый run;07 по умолчанию `RUN_TRAINING=False` и сравнивает все completed50. Конфиг готов для Kaggle, обучение не выполнялось; можно сохранить нынешнийpartial и завершить поиск без этого запуска.

Consistency clean/aug (например [AugMix](https://arxiv.org/abs/1912.02781), [авторский код](https://github.com/google-research/augmix)) также содержательнее очередного уровняblur: цель — согласованность предсказаний разных повреждённых views одного train-кадра. Но это новый, более дорогой training objective, и он не гарантирует исправления чётких семантически похожих зданий. В этой серии он не измерен и не включён как второй обязательный запуск; не подменяем отрицательные результаты новой пачкой методов.

Оригинальные journal, raw, configs/weights/predictions completed runs не изменены. Производные таблицы/изображения и один CosFace validation diagnostic сохранены отдельно; обучения и test-fitting не было.
