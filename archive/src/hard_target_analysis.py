"""Read-only val comparisons and quality case galleries for paired experiments."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from archive.src.training.predict import load_probabilities
from archive.src.error_analysis import class_metrics, confusion_pairs, image_gallery


def compare_shift_runs(root, run, baselines, classes, raw_dir):
    root, raw_dir = Path(root), Path(raw_dir)
    candidate = load_probabilities(root / 'artifacts/predictions' / run / 'val_probabilities.npz')
    frame = candidate.table(include_scores=True).rename(columns={'label': run})
    summaries = []
    for name in [run, *baselines]:
        if name != run:
            path = root / 'artifacts/predictions' / name / 'val_probabilities.npz'
            if not path.is_file():
                print(f'Missing baseline probabilities: {name}')
                continue
            baseline = load_probabilities(path)
            if set(baseline.ids) != set(candidate.ids) or set(baseline.class_labels) != set(candidate.class_labels):
                raise ValueError('Comparison requires the same val IDs and class labels')
            old = baseline.table().set_index('id').loc[frame.id].reset_index()
            if not np.array_equal(old.true_label, frame.true_label):
                raise ValueError('Comparison targets differ')
            frame[name] = old.label.to_numpy()
        correct = frame[name].eq(frame.true_label)
        new_correct = frame[run].eq(frame.true_label)
        summaries.append({'run': name, 'accuracy': correct.mean(), 'errors': int((~correct).sum()),
                          'fixed_by_new': int((~correct & new_correct).sum()),
                          'broken_by_new': int((correct & ~new_correct).sum())})
    # Image quality proxies are diagnostics, NOT ground-truth degradation labels.
    # Small-object visibility cannot be inferred reliably from image-wide stats.
    quality = []
    for image_id in frame.id:
        path = (raw_dir / 'val' / image_id).resolve()
        if not path.is_relative_to((raw_dir / 'val').resolve()):
            raise ValueError('Image ID escapes val directory')
        with Image.open(path) as source:
            gray = np.asarray(source.convert('L'), dtype=float)/255
            smooth = np.asarray(source.convert('L').filter(ImageFilter.GaussianBlur(1)), dtype=float)/255
        quality.append({'id': image_id, 'contrast_std': gray.std(), 'brightness': gray.mean(),
                        'high_frequency': np.abs(gray-smooth).mean()})
    frame = frame.merge(pd.DataFrame(quality), on='id', validate='one_to_one')
    return pd.DataFrame(summaries), frame


def shift_error_report(frame, run, classes, raw_dir, *, page_size=12, quality_limit=12, manual_ids=()):
    """Return metrics, show quality-related mistakes and manually chosen small objects."""
    from IPython.display import display
    display(class_metrics(frame, run, classes))
    display(confusion_pairs(frame, run, classes).head(15))
    names = classes.set_index('label').class_name.to_dict()
    wrong = frame[frame[run] != frame.true_label]
    def gallery(subset, title):
        rows = [(row['id'], f"true: {names[row['true_label']]}\npred: {names[row[run]]}")
                for _, row in subset.iterrows()]
        image_gallery(rows, Path(raw_dir)/'val', title, columns=4, page_size=page_size)
    gallery(wrong.nsmallest(quality_limit, 'high_frequency'), 'Errors: low-detail / blur proxy (inspect visually)')
    gallery(wrong.nsmallest(quality_limit, 'contrast_std'), 'Errors: low-contrast proxy (inspect visually)')
    gallery(wrong.nsmallest(quality_limit, 'brightness'), 'Errors: darkest cases')
    if manual_ids:
        gallery(frame[frame.id.isin(manual_ids)], 'Manually selected small-object / quality cases')
    target_names = {'tennis_court', 'basketball_court', 'church', 'palace'}
    labels = classes.loc[classes.class_name.isin(target_names), 'label']
    gallery(wrong[wrong.true_label.isin(labels)].head(quality_limit),
            'Errors: classes often involving local details (not automatic small-object labels)')
    for baseline in [col for col in frame.columns if col.startswith('resnet50') and col != run]:
        old_ok, new_ok = frame[baseline].eq(frame.true_label), frame[run].eq(frame.true_label)
        display(frame.loc[old_ok != new_ok, ['id', 'true_label', baseline, run, 'brightness', 'contrast_std', 'high_frequency']])
        gallery(frame[~old_ok & new_ok].head(quality_limit), f'Fixed vs {baseline}')
        gallery(frame[old_ok & ~new_ok].head(quality_limit), f'Broken vs {baseline}')
