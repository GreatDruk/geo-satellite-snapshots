"""Read-only analysis of saved validation predictions and train/val images."""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def load_predictions(project_root, experiment_ids):
    """Align every experiment to raw val IDs and verify labels before comparison."""
    root = Path(project_root)
    classes = pd.read_csv(root / "data/raw/classes.csv")
    labels = classes.label.tolist()
    if classes.label.duplicated().any() or classes.class_name.duplicated().any():
        raise ValueError("Class labels and names must be unique")
    frame = pd.read_csv(root / "data/raw/val.csv", dtype={"id": str})
    if frame.empty or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("Validation IDs must be nonempty and unique")
    frame = frame.rename(columns={"label": "true_label"}).set_index("id")
    if not frame.true_label.isin(labels).all():
        raise ValueError("Unknown validation labels")
    if not experiment_ids or len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("Choose distinct experiment IDs")
    for exp_id in experiment_ids:
        pred = pd.read_csv(root / "artifacts/predictions" / exp_id / "val.csv",
                           dtype={"id": str})
        if pred.id.isna().any() or pred.id.duplicated().any() or set(pred.id) != set(frame.index):
            raise ValueError(f"Prediction IDs do not match val: {exp_id}")
        pred = pred.set_index("id").loc[frame.index]
        if not pred.label.isin(labels).all():
            raise ValueError(f"Unknown predicted labels: {exp_id}")
        if "true_label" in pred and not pred.true_label.equals(frame.true_label):
            raise ValueError(f"Saved ground truth does not match raw val: {exp_id}")
        frame[exp_id] = pred.label
    return frame, classes


def model_comparison(frame, experiment_ids):
    rows = []
    for name in experiment_ids:
        correct = frame[name].eq(frame.true_label)
        rows.append({"experiment": name, "correct": int(correct.sum()),
                     "errors": int((~correct).sum()), "accuracy_pct": 100 * correct.mean()})
    return pd.DataFrame(rows).set_index("experiment")


def confusion_counts(frame, experiment, classes):
    return pd.crosstab(frame.true_label, frame[experiment]).reindex(
        index=classes.label, columns=classes.label, fill_value=0)


def class_metrics(frame, experiment, classes):
    counts = confusion_counts(frame, experiment, classes).to_numpy()
    support, predicted = counts.sum(axis=1), counts.sum(axis=0)
    tp = np.diag(counts)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(recall), where=(precision + recall) > 0)
    return pd.DataFrame({"class_name": classes.class_name.to_numpy(), "support": support,
                         "errors": support - tp, "false_positives": predicted - tp,
                         "recall_pct": 100 * recall, "precision_pct": 100 * precision,
                         "f1_pct": 100 * f1}).sort_values("errors", ascending=False,
                                                          kind="stable").reset_index(drop=True)


def confusion_pairs(frame, experiment, classes):
    names = classes.set_index("label").class_name.to_dict()
    wrong = frame.loc[frame[experiment].ne(frame.true_label)]
    pairs = wrong.groupby(["true_label", experiment]).size().reset_index(name="count")
    pairs["true_class"] = pairs.true_label.map(names)
    pairs["predicted_class"] = pairs[experiment].map(names)
    return pairs[["true_class", "predicted_class", "count"]].sort_values(
        "count", ascending=False, kind="stable").reset_index(drop=True)


def plot_confusion(frame, experiment, classes):
    counts = confusion_counts(frame, experiment, classes).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    support = counts.sum(axis=1, keepdims=True)
    recall = np.divide(100 * counts, support, out=np.zeros_like(counts, dtype=float),
                       where=support > 0)
    for ax, values, title in zip(axes, [counts, recall], ["Counts", "Row percentages (%)"]):
        ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes.class_name, rotation=90, fontsize=8)
        ax.set_yticks(range(len(classes)), classes.class_name, fontsize=8)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        for i, j in zip(*np.nonzero(values)):
            text = str(int(values[i, j])) if title == "Counts" else f"{values[i, j]:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if values[i, j] > values.max() * 0.55 else "black")
    fig.suptitle(experiment)
    fig.tight_layout()
    plt.show()
    return fig


def image_gallery(rows, image_dir, title, columns=4, page_size=12):
    """Show all requested original RGB images in bounded pages, without augmenting."""
    if columns < 1 or page_size < 1:
        raise ValueError("Positive columns and page_size required")
    rows = list(rows)
    if not rows:
        print(f"{title}: no images")
        return []
    directory = Path(image_dir).resolve()
    figures = []
    for start in range(0, len(rows), page_size):
        page = rows[start:start + page_size]
        fig, axes = plt.subplots(math.ceil(len(page) / columns), columns,
                                 figsize=(columns * 3.5, math.ceil(len(page) / columns) * 4.7),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (image_id, caption) in zip(axes.flat, page):
            path = (directory / image_id).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Image ID escapes split directory")
            with Image.open(path) as source:
                ax.imshow(source.convert("RGB"), interpolation="nearest")
            ax.set_title(f"{image_id}\n{caption}", fontsize=9)
        fig.suptitle(f"{title} — {start + 1}–{start + len(page)} / {len(rows)}")
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.91,
                            hspace=0.55, wspace=0.08)
        plt.show()
        figures.append(fig)
    return figures


def show_errors(frame, experiment, classes, raw_dir, true_class=None,
                predicted_class=None, comparison=None):
    names = classes.set_index("label").class_name.to_dict()
    selected = frame.loc[frame[experiment].ne(frame.true_label)]
    if true_class is not None:
        selected = selected.loc[selected.true_label.map(names).eq(true_class)]
    if predicted_class is not None:
        selected = selected.loc[selected[experiment].map(names).eq(predicted_class)]
    rows = []
    for image_id, row in selected.iterrows():
        caption = f"True: {names[row.true_label]}\nPred: {names[row[experiment]]}"
        if comparison is not None:
            caption += f"\nOther: {names[row[comparison]]}"
        rows.append((image_id, caption))
    return image_gallery(rows, Path(raw_dir) / "val", f"Errors: {true_class or 'all'}")


def show_train_examples(raw_dir, classes, class_names, examples_per_class=6, seed=42):
    """Deterministic random train examples, not cherry-picked nearest neighbours."""
    raw = Path(raw_dir)
    train = pd.read_csv(raw / "train.csv", dtype={"id": str})
    lookup = classes.set_index("class_name").label.to_dict()
    figures = []
    for name in class_names:
        selected = train.loc[train.label.eq(lookup[name])]
        selected = selected.sample(min(examples_per_class, len(selected)), random_state=seed)
        figures.extend(image_gallery([(r.id, name) for r in selected.itertuples()],
                                      raw / "train", f"Train reference: {name}", columns=3))
    return figures


def paired_comparison(frame, primary, other):
    first = frame[primary].eq(frame.true_label)
    second = frame[other].eq(frame.true_label)
    return pd.Series({"both_correct": int((first & second).sum()),
                      "both_wrong": int((~first & ~second).sum()),
                      "primary_only_correct": int((first & ~second).sum()),
                      "other_only_correct": int((~first & second).sum()),
                      "different_predictions": int(frame[primary].ne(frame[other]).sum()),
                      "oracle_accuracy_pct_NOT_ensemble": 100 * (first | second).mean()})


def specialist_routing_audit(frame, experiment, classes, groups):
    """Oracle ceiling for disjoint groups routed solely by current predicted class.

    True labels are used to measure potential, never to define an inference route.
    This is NOT the performance of a trained expert or ensemble.
    """
    labels = classes.set_index("class_name").label.to_dict()
    seen, rows = set(), []
    incorrect = frame[experiment].ne(frame.true_label)
    for name, class_names in groups.items():
        group = {labels[item] for item in class_names}
        if seen & group:
            raise ValueError("Routing groups must be disjoint")
        seen |= group
        routed = frame[experiment].isin(group)
        true_inside = frame.true_label.isin(group)
        rows.append({"group": name, "routed_images": int(routed.sum()),
                     "correct_answers_at_risk": int((routed & ~incorrect).sum()),
                     "within_group_errors_potentially_fixable": int((routed & true_inside & incorrect).sum()),
                     "outside_group_images_need_reject": int((routed & ~true_inside).sum()),
                     "group_errors_missed_by_route": int((~routed & true_inside).sum())})
    return pd.DataFrame(rows).set_index("group")
