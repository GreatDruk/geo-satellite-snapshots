# ResNet50 + LGFFE

Запуск: [12_resnet50_lgffe.ipynb](../notebooks/12_resnet50_lgffe.ipynb). Конфиг: [resnet50_lgffe_domain_shift_stronger.json](../configs/resnet50_lgffe_domain_shift_stronger.json). ID: `resnet50_lgffe_domain_shift_stronger_001`. Реальное обучение ещё не выполнено; test не используется.

## Архитектура

Источник: Lv et al., [An End-to-End Local-Global-Fusion Feature Extraction Network for Remote Sensing Image Scene Classification](https://www.mdpi.com/2072-4292/11/24/3006), Remote Sensing 2019, DOI 10.3390/rs11243006, разделы 3.1–3.3, уравнения 6–9.

Применён именно GRU-based extractor: последовательность пространственных регионов проходит через GRU; общий для регионов Linear преобразует hidden states в скалярные scores, затем ReLU и Linear между scores всех регионов уточняют веса. Softmax нормируется по регионам. Local feature — взвешенная сумма **исходных CNN features**, а не GRU hidden states. GAP и local features объединяются; два Linear слоя классифицируют полученный вектор. QKV, Transformer, self-attention или CBAM нет.

При batch B и входе 256×256:

| Тензор | Shape |
|---|---|
| ResNet50 layer4 до avgpool/fc | B × 2048 × 8 × 8 |
| Региональная последовательность, построчный порядок | B × 64 × 2048 |
| GRU states, однонаправленная, один слой | B × 64 × 512 |
| Общий Linear(512,1) → ReLU | B × 64 |
| Linear(64,64) → spatial softmax | B × 64 |
| GAP / local weighted sum | каждый B × 2048 |
| Concat | B × 4096 |
| Linear(4096,512) → LayerNorm → ReLU → Dropout(0.3) | B × 512 |
| Linear(512,20), сырые logits | B × 20 |

Attention использует фиксированную пространственную сетку. Размер определяется из image_size при сборке и сохраняется в checkpoint config; нельзя подавать другой размер изображения без соответствующей конфигурации и новых весов attention refinement. Диагностический `model.forward_details(images)` возвращает attention, feature map и обе ветви; обычный `model(images)` возвращает только logits.

## Осознанные адаптации статьи

Статья объясняет алгоритм на VGG feature map 14×14×512. В описании второго FC есть несогласованность между числом регионов 196 и указанной размерностью 512; здесь его размер определяется фактическим числом регионов (64). Сохранены GRU, общий региональный score, refinement, softmax и weighted sum. Для требуемого ResNet50 layer4 используются исходные 2048 каналов без дополнительной проекции и без изменения stride backbone.

В нашей новой голове LayerNorm заменяет BatchNorm классификатора из статьи: это позволяет сохранить frozen backbone BN и избежать обучения/заморозки новых BN statistics после случайной инициализации. Размеры GRU и hidden head по 512 — явные настройки этого эксперимента. Статья также использует SVM поверх признаков при оценке; здесь по запросу используется единая CE/logits голова, совместимая с текущим pipeline. Это реализация механизма LGFFE под заданные условия, а не точное воспроизведение всех настроек и протокола оценки статьи.

## Обучение и сравнение

ImageNet initialization: `ResNet50 IMAGENET1K_V2`. Первые 3 эпохи обучаются все новые GRU/attention/head параметры; далее добавляются layer3+layer4. Backbone BN всегда eval, running statistics не обновляются; affine weights в выбранных блоках ведут себя так же, как в текущем stronger50. Layer1/layer2/conv1 остаются замороженными. Новая ветвь целиком входит в head optimizer group.

AdamW: head LR 5e-4, backbone LR 5e-5, weight decay 1e-3. Cosine T_max=25, eta_min=1e-6. 25 эпох, batch32, seed42, ordinary CrossEntropyLoss без smoothing. Stronger domain_shift photometric/corruption settings скопированы без изменений. Геометрия заменена на uniform D4 из 8 точных transpose операций, без crop/интерполяции при повороте. Val не аугментируется, test не загружается.

Baseline: сохранённая `resnet50_partial_domain_shift_stronger_001` без TTA. Notebook сравнивает Accuracy, исправленные/новые/общие val ошибки и показывает изменившиеся ответы. Изменены **и extractor, и train geometry**; результат одного запуска оценивает этот пакет, не отдельный вклад LGFFE. Подтверждение происхождения нашего датасета из NWPU-RESISC45 само по себе не переносит опубликованные scores на наши классы, split и искажения.

## Запуск и артефакты

На Kaggle загрузить обновлённый src, новый конфиг и notebook. Настроить `RAW_DIR`, при необходимости batch size; включить интернет для первоначальной загрузки torchvision V2 weights либо подготовить cache. В notebook установить `RUN_TRAINING=True`; остальные модели не запускаются. Synthetic preflight выполнится прежде обучения. Неиспользованный ID защищает от перезаписи.

После полного обучения общий pipeline сохранит:

- `artifacts/models/<exp_id>/best.pt` и `last.pt`;
- `artifacts/logs/<exp_id>/config.json`, `metadata.json`, `training.log`, `history.csv`, `learning_curves.png`, `summary.json`;
- `artifacts/predictions/<exp_id>/val.csv` и `val_probabilities.npz`;
- одну обычную training-запись в `experiments.csv`.

`load_checkpoint` восстанавливает LGFFE без скачивания pretrained weights. Обычный inference и существующий D4 TTA поддерживаются через тот же logits interface. Старые конфиги, веса и артефакты не меняются. LGFFE synthetic checks не создают training-запись или реальные run artifacts.

## Выполненные проверки

`python -m unittest tests.test_lgffe tests.test_transfer tests.test_augmentations -v`: 22 теста, 21 пройден, один CUDA-only тест пропущен на CPU. Настоящая torchvision ResNet50 проверена на двух synthetic 256×256 изображениях: forward/backward, нормировка attention, градиенты GRU/attention, обновление головы во время warmup и layer4 после него, неизменность ранних весов и всех BN running buffers. Проверены восемь различных D4 pixel permutations и выключение D4 на val. Синтетический мини-run во временной директории проверил общий цикл, best/last checkpoints, history/графики, val NPZ, journal и восстановленный inference с D4 без test access. ImageNet V2 download и реальный CUDA training этим CPU-проходом не проверялись.
