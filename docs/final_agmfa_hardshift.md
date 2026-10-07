# Две финальные corrected AGMFA HardShift

Запуск: `notebooks/16_final_agmfa_hardshift.ipynb`.
Configs: `resnet50_agmfa_hardshift_final_train.json` и
`resnet50_agmfa_hardshift_final_train_val.json`.
Новые exp IDs: `resnet50_agmfa_hardshift_final_train_001` и
`resnet50_agmfa_hardshift_final_train_val_001`.

Обе модели обучаются заново от ResNet50 ImageNet V2, с новой AGMFA head,
без использования checkpoint успешного B или варианта C. Архитектура corrected
AGMFA прежняя: attention_source conv2, partial layer3/4, frozen BN, warmup3.
Seed42, AdamW backbone LR5e-5/head LR5e-4, WD0.001, input256, batch16 объектов.
Фиксированы **22 эпохи**, cosine T_max22/eta_min1e-6. Единственный итоговый
checkpoint — последняя эпоха; best accuracy/best loss selection отсутствует.

Сохранённая история B подтверждает основание выбора: epoch14 — 675/699,
val loss0.130131, hard acc0.909982, JS0.045168; epoch22 — 674/699,
val loss0.109995, hard acc0.946389, JS0.032884. Эти значения относятся к
старому 24-эпоховому cosine run. Новые свежие обучения с cosine22 не обязаны
повторить его цифры. 22 эпохи фиксируются заранее, будущая val их не выбирает.

## Различие происхождения объектов

Source train: прежние paired clean+hard views с общим random D4,
6 обязательных profiles `(index+epoch)%6`, параметры из seed/epoch/index.
Loss на **каждый объект**: `0.5 CE(clean)+0.5 CE(hard)+3 JS`.

Бывший val в refit: только full-frame RGB/resize256, один uniform D4, tensor,
ImageNet normalization, один forward и обычный CE. Hard-view не создаётся,
profile ID отсутствует; placeholders/повторный forward target не используются.
Никаких синтетических blur/noise/color cast на бывшем val.

Происхождение и локальный index сохраняются отдельно, даже при совпадающих
ID в разных split. Обычный shuffled объединённый dataset используется без
oversampling/drop_last: в каждой эпохе каждый train/val объект ровно один раз.
Shared epoch обновляется у persistent workers для обоих split.

Loss для mixed batch:
`(sum(source_object_losses)+2*sum(target_CE))/(n_source+2*n_target)`.
Batch16 объектов даёт от16 до32 views; для чистого source batch —32.
Source-only и target-only batches корректно поддерживаются.
Epoch total loss агрегируется по той же взвешенной массе; clean/hard метрики
усредняются по source, target CE/Accuracy — отдельно по target.
Эффективный target вклад: `2*699/(4999+2*699) ≈21.85%`.

Training loop проверяет, что каждый source/target index встретился ровно один
раз, фактические source profile counts совпадают с циклическим расписанием,
forward images = 2*n_source+n_target. 22 эпохи —3 полных profile cycles+4 эпохи.

## Training и checkpoints

Флаги по умолчанию: RUN_TRAIN_ONLY=False, RUN_TRAIN_VAL=False, RUN_TEST=False.
Укажите ROOT, RAW_DIR, DEVICE. На Kaggle используйте auto; при нехватке памяти
уменьшите batch16 до8 до первого запуска. Изменение batch не меняет profile rule.

- Train-only: исходная deterministic val используется только для отчёта;
  scheduler зависит от epoch. После epoch22 сохраняются final.pt и его
  byte-for-byte копия last.pt, val.csv и val_probabilities.npz последних весов.
- Train+val: validation loader не создаётся; evaluate_model и validation
  selection не вызываются. После epoch22 сохраняется final.pt (также last.pt).
  Val accuracy в summary/experiments пустая, псевдо-held-out метрика не вычисляется.

В `artifacts/logs/<exp_id>/`: config, metadata, history.csv, profile_counts.csv,
training.log, learning_curves.png и summary. Графики разделяют source clean/hard
CE/Accuracy, JS, target CE/Accuracy, общий loss и LR; только train-only имеет val
curves. Успешное обучение добавляет новую запись experiments.csv.
Существующие run directories и journal IDs защищены от перезаписи.

## Ровно две submissions

После завершения обеих моделей (или загрузки их весов) включите RUN_TEST=True.
Функция make_final_submissions отдельно загружает train-only last.pt и
train+val final.pt; проверяет epoch22, split, recipe, class order и normalization.
Обычный original preprocessing и один forward на test, TTA/ансамблей нет.
CPU inference после обучения на CUDA поддерживается; выбор device не меняет recipe.

CSV в `ROOT/submissions/`:

- `submission_agmfa_hardshift_train.csv`
- `submission_agmfa_hardshift_train_val.csv`

В `artifacts/predictions/<exp_id>/`: test.csv/test_probabilities.npz.
В `artifacts/logs/<exp_id>/`: test_inference.json с путём checkpoint и аудитом.
Все NPZ/CSV имеют порядок sample_submission_example.csv. Проверяются ровно
8299 unique IDs, точное совпадение множеств IDs, отсутствие дублей/пропусков,
совпадение labels с argmax probabilities. Только этот явный inference читает
test/sample submission; обучение их не загружает. Файлы не отправляются на Kaggle.
Две отдельные inference записи `<training_exp_id>_test` добавляются в журнал,
прежние training/submission записи не редактируются.

## Проверки

Новые тесты используют synthetic данные и маленькую модель во временной папке:
exact weighted objective/gradients, source-only/target-only batches, отсутствие
hard profiles у target, persistent worker epoch updates, once-per-epoch sampling,
оба 22-эпоховых цикла и сохранение **последнего** checkpoint даже при более высокой
ранней val Accuracy. Refit тест запрещает любые evaluate_model calls.
Отдельно экспортируются два synthetic submission на 8299 IDs с намеренно
разным test/sample order; проверяются порядок, argmax, no TTA и no overwrite.

В текущем локальном окружении .venv Python path отсутствует, а torchvision
несовместим с torch. Тесты нового цикла работают через доступный Python и
уже имеющийся torch, с synthetic model_factory; реальный pretrained AGMFA
forward повторно локально не проверялся. Библиотеки не устанавливались,
исходная реализация AGMFA и все прежние конфигурации/артефакты не менялись.
Реальные обучения/test inference автоматически не запускались.
