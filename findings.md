# Findings

## Известные сведения о соревновании

- Задача: классификация спутниковых изображений, 20 классов.
- Train: 4999 изображений.
- Val: 699 изображений.
- Test: 8299 изображений.
- Train содержит относительно чистые изображения.
- Val/test содержат реальные артефакты и искажения.
- Аугментации и устойчивость к domain shift — важное направление экспериментов.
- Метрика: Accuracy.
- Обязательно сравнить MLP, CNN и хотя бы одну pretrained-модель.
- Разрешённые pretrained-архитектуры: AlexNet, VGG, ResNet, Inception (исправление ошибки в правилах).
- Transformers запрещены.
- Test нельзя использовать для fitting preprocessing, feature selection, representations, clustering или моделей.
- Финальный pipeline должен быть воспроизводимым.

## Data observations

## Models

## Augmentations

## Error analysis

## Final solution
