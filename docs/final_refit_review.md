# Проверка final train+val refit и четыре тестовых сабмита

**Обновление:** все четыреtest submissions уже выполнены пользователем наKaggle, suite `submissions_001`, и прошлиполнуюпроверку файлов/журнала/520chunks. [Результат](test_submissions_review.md). Ниже сохранена инструкция ипредыдущаяпроверкаrefit; утверждение оещёне выполненномtest inference относится кпрошломуэтапу.

Оба запуска проверены по config/history/summary/metadata/journal/actual checkpoint. [Машиночитаемый итог](final_refit_audit.csv). Признаков ошибки выполнения не найдено.

| Запуск | Эпох | Изображений каждую эпоху | Последняя train Accuracy | Последний train loss | Минут |
|---|---:|---:|---:|---:|---:|
| ResNet50 final | 20 | 5698 | 99.6139% | 0.012952 | 15.61 |
| ResNet101 final | 21 | 5698 | 99.7543% | 0.006069 | 30.37 |

Проверено: complete status, train4999+val699, seed42, partial/frozenBN/warmup3/stronger-аугментации/optimizer соответствуют presets; каждый epoch finite, последовательность epochs полная, cosine LR обеих групп соответствует T_max25 иeta_min1e-6. Метаданные не содержат test fitting; исходные CSV hashes совпадают. В experiments.csv ровно по одной training-записи, val_accuracy пустая, время согласуется сsummary.

SHA256 actual final.pt совпадает с сохранёнными метаданными:50 `0ba6befb21249ad3312ccd07d5cd597c0c9eace1ba6fdff3ed16e483eab9558a`;101 `7f75a1110cb6f3448f37523134891c1af7c2934d04b45276bd9cebe382d1a500`. Epoch/config/classes совпадают, tensors finite. Оба checkpoint загружаются существующимload_checkpoint и проходят synthetic CPU forward256×256 с выходом1×20. Реальные val-изображения для оценки refit не прогонялись.

Train Accuracy — показатель выполнения обучения, не доказательство улучшения generalization. Небольшие поздние колебания train loss/Accuracy при случайных аугментациях не выглядят срывом. Размер выигрыша наtest пока неизвестен; без истинныхtest labels локальнуюAccuracy вычислить нельзя.

## Четыре фиксированных рецепта

1. ResNet101, обученная только наtrain: исходный best.pt, безTTA.
2. ResNet101, обученная наtrain+val: новый final.pt, безTTA.
3. Прежний лучший ансамбльtrain: ResNet101 best.pt none + stronger50 best.pt D4.
4. Тот же ансамбльtrain+val: ResNet101 final.pt none + ResNet50 final.pt D4.

Ensemble=0.5*p101+0.5*p50D4. D4=mean probabilities восьми orientations полного кадра, не crops. Вес50D4 остаётся0.5, а не8/9. CosFace/mining/reranker не добавляются. Это достаточный набор для сравнения эффекта объединённых данных отдельно уsingle101 иensemble; одиночная50 не обязательна.

## Запуск

**notebooks/10_test_submissions.ipynb**, `src/submission_suite.py`. ПроверитьPROJECT_ROOT/RAW_DIR/четыреCHECKPOINT_PATHS, поставитьRUN_TEST=True и выполнить notebook наKaggle GPU. По умолчаниюFalse, чтобы случайно не запускать длительный полныйCPU inference. CPU поддерживается. Повторныйaudit можно включитьRUN_AUDIT=True при наличии training logs/configs.

Каждыйcheckpoint прогоняется один раз: дваnone101 и дваD450, суммарно18 forwards наtest-объект. Кэш101 используется дляsingle иensemble. Batch-кэши позволяют продолжить послеобрыва с тем жеSUITE_ID/weights/CSV/batch_size. Завершённый повтор не дублируетexperiment rows; другойcheckpoint/CSV/protocol с темжеID отвергается. Существующие training-записи не редактируются.

Выходы: четыреCSV вsubmissions/, probabilities участников/рецептов иbatch-кэши вartifacts/predictions/<suite>/; SHA256/configs/epochs/прогресс/времяinference иsubmissions.csv вartifacts/logs/<suite>/. В experiments.csv четыре новыеTEST INFERENCE ONLY строки: submission_id заполнен, val_accuracy/train_time_min/public_score пустые. Времяinference хранится отдельно. CSV имеет толькоid,label, уникальныеIDs и порядокsample_submission_example; в текущих данных8299строк.

НаKaggleCSV загружаются отдельно пользователем; полученныйpublic_score вписывается вновыеsubmission-строки. Автоматической внешней загрузки нет. Никакой fitting/feature selection поtest. На этом шаге проведены synthetic CPU tests иdry-run notebook; полныйreal test inference ещё не запускался.
