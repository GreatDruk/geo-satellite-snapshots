# Проверка четырёх test submissions

Suite `submissions_001` завершён корректно. CUDA, batch64, seed42; полный inference занял **14.06мин**. Повторного обучения или fitting поtest нет. Четыре CSV сохранены вsubmissions/.

Проверка: исходныеtest/classes/sample CSV и все четыре actual checkpoint совпадают сSHA256 вprotocol. Использованы нужныеtrain best.pt иtrain+val final.pt:101 epoch21 безTTA,50 epoch20 сD4. Есть полные130 batches каждого участника; проверены **все520 batch-кэшей** — порядокIDs, class labels и probabilities точно совпадают с объединённымиNPZ. Массивыfinite, probabilities в[0,1], суммы строк≈1; test targets отсутствуют.

Вероятностиsingle точно совпадают с соответствующей101. Обеensemble NPZ точно воспроизводятся как0.5*p101+0.5*p50D4, сfloat64 накоплением иfloat32 сохранением исходного кода. Неусредняемыеlogits/голоса не использовались. Каждыйsubmission содержит8299 уникальныхIDs, толькоid,label, допустимые20labels, безпропусков; порядок совпадает сsample. Каждыйlabel точно равенargmax сохранённыхprobabilities; hashes CSV согласуются сmetadata, индивидуальнымиJSON иsubmissions.csv.

В experiments.csv ровночетыре новыеTEST INFERENCE ONLY записи,submission_id корректны, seed42. val_accuracy иtrain_time_min пустые поназначению; public_score внесён по скриншоту пользователя: 0.94457 / 0.94767 / 0.95111 / 0.95370 для single train / single train+val / ensemble train / ensemble train+val. Training/refit записи остаются отдельно. Локальной test Accuracy нет. [Анализ public scores и ответов](test_prediction_analysis_review.md).

| Участник | TTA | Времяinference, мин |
|---|---|---:|
| 101 train | none | 2.02 |
| 50 train | D4 | 5.07 |
| 101 train+val | none | 1.72 |
| 50 train+val | D4 | 5.17 |

Refit изменил313 предсказанийsingle101 и197предсказанийensemble. Это подтверждает, что используютсяразныевеса/результаты, но не показывает, сколькоисправлений илирегрессий — безtest labels так утверждатьнельзя. Добавление50D4 меняет134ответаtrain101 и142ответаrefit101; это также неAccuracy.

[Полный аудит](test_submissions_audit.json), воспроизводимаяпроверка: `python docs/audit_test_submissions.py`. Скрипт только читает пользовательскиеартефакты и сохраняет производныйотчёт; не перезапускаетinference, не меняетjournal/submissions/weights/raw.
