from archive.src.utils import raw_data_dir
"""Read-only analysis of saved validation predictions and train/val images."""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def validation_probabilities(project_root, experiment_id, *, raw_dir=None,
                             probabilities_path=None, device="auto", batch_size=16,
                             num_workers=0):
    """Load full val probabilities; if absent infer best.pt on val only and cache.

    Existing arrays are checked against current val IDs, targets and class labels.
    An invalid cache is an error, never a reason to silently replace it.
    """
    from archive.src.training.predict import load_probabilities, save_probabilities

    root = Path(project_root).resolve()
    raw = Path(raw_dir) if raw_dir is not None else raw_data_dir(root)
    if not raw.is_absolute():
        raw = root / raw
    if not isinstance(experiment_id, str) or Path(experiment_id).name != experiment_id or experiment_id in {"", ".", ".."}:
        raise ValueError("Use an experiment directory name, not a path")
    path = (Path(probabilities_path) if probabilities_path is not None else
            root / "artifacts/predictions" / experiment_id / "val_probabilities.npz")
    if not path.is_absolute():
        path = root / path
    classes = pd.read_csv(raw / "classes.csv")
    truth = pd.read_csv(raw / "val.csv", dtype={"id": str})
    if truth.id.isna().any() or truth.id.duplicated().any() or classes.label.duplicated().any():
        raise ValueError("Validation IDs and class labels must be unique and nonempty")
    generated = False
    if path.exists():
        result = load_probabilities(path)
        print(f"Loaded validation probabilities: {path}")
    else:
        from archive.src.data import load_validation
        from archive.src.experiment import load_checkpoint
        from archive.src.training.predict import probability_predictions
        from archive.src.utils import get_device, set_seed

        if batch_size < 1 or num_workers < 0:
            raise ValueError("Positive batch_size and nonnegative num_workers required")
        selected_device = get_device(device)
        checkpoint_path = root / "artifacts/models" / experiment_id / "best.pt"
        print(f"No probabilities found; validation-only inference on {selected_device}: {checkpoint_path}")
        set_seed(42, deterministic=True, warn_only=True)
        model, checkpoint = load_checkpoint(checkpoint_path, selected_device, inference_only=True)
        config = dict(checkpoint["config"]["data"])
        # Use the notebook's explicit data location, not a machine-specific saved path.
        config.update(batch_size=batch_size, num_workers=num_workers,
                      pin_memory=selected_device.type == "cuda")
        if num_workers == 0:
            config.pop("prefetch_factor", None)
            config["persistent_workers"] = False
        loader, _ = load_validation(raw, config, checkpoint["config"].get("seed", 42))
        def progress(batch, total):
            if batch == 1 or batch % 10 == 0 or batch == total:
                print(f"Validation inference: {batch}/{total} batches", flush=True)
        result = probability_predictions(model, loader, selected_device,
                                         checkpoint["class_labels"], tta="none", progress=progress)
        generated = True
        del model, loader, checkpoint
    result.validate()
    if set(result.ids) != set(truth.id) or set(result.class_labels) != set(classes.label):
        raise ValueError("Saved probabilities do not match validation IDs/classes")
    if result.true_labels is None or not np.array_equal(
            result.true_labels, truth.set_index("id").loc[result.ids, "label"].to_numpy()):
        raise ValueError("Saved probability targets do not match raw validation")
    if generated:
        save_probabilities(result, path)
        print(f"Saved validation probabilities: {path}")
    return result, classes


def probability_cases(predictions, classes, *, selection="errors", limit=None, top_k=5):
    """Rank errors by confidence or all val images by smallest top1-top2 gap.

    Includes the true-class probability/rank even when absent from top-k.
    Returns one row per image; no ground truth is used for uncertainty ranking.
    """
    predictions.validate()
    if predictions.true_labels is None:
        raise ValueError("Validation ground truth is required")
    if selection not in {"errors", "uncertain"} or not isinstance(top_k, int) or not 2 <= top_k <= len(predictions.class_labels):
        raise ValueError("Choose errors/uncertain and top_k between 2 and class count")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be None or a positive integer")
    names = classes.set_index("label").class_name.to_dict()
    if set(names) != set(predictions.class_labels):
        raise ValueError("Class name mapping does not match probabilities")
    p = predictions.probabilities
    order = np.argsort(-p, axis=1, kind="stable")
    rows = []
    for i, image_id in enumerate(predictions.ids):
        true_label = predictions.true_labels[i]
        predicted = predictions.class_labels[order[i, 0]]
        true_index = int(np.flatnonzero(predictions.class_labels == true_label)[0])
        rows.append({"id": image_id, "true_class": names[true_label],
                     "predicted_class": names[predicted], "correct": bool(predicted == true_label),
                     "confidence": float(p[i, order[i, 0]]),
                     "second_probability": float(p[i, order[i, 1]]),
                     "margin": float(p[i, order[i, 0]] - p[i, order[i, 1]]),
                     "true_probability": float(p[i, true_index]),
                     "true_rank": int(np.flatnonzero(order[i] == true_index)[0]) + 1,
                     "top_classes": [names[predictions.class_labels[j]] for j in order[i, :top_k]],
                     "top_probabilities": [float(p[i, j]) for j in order[i, :top_k]]})
    table = pd.DataFrame(rows)
    if selection == "errors":
        table = table.loc[~table.correct].sort_values("confidence", ascending=False, kind="stable")
    else:
        table = table.sort_values("margin", ascending=True, kind="stable")
    return table.head(limit) if limit is not None else table


def plot_probability_cases(cases, image_dir, title, *, page_size=6):
    """Each case has an original image beside top-k horizontal probability bars."""
    if not isinstance(page_size, int) or page_size < 1:
        raise ValueError("page_size must be positive")
    if cases.empty:
        print(f"{title}: no images")
        return []
    directory = Path(image_dir).resolve()
    def probability_text(value):
        if 0 < value < .0001:
            return f"{value * 100:.2e}%"
        return f"{value:.6%}" if value >= .9999 else f"{value:.2%}"
    figures = []
    for start in range(0, len(cases), page_size):
        page = cases.iloc[start:start + page_size]
        fig, axes = plt.subplots(len(page), 2, figsize=(12, 3.5 * len(page)),
                                 squeeze=False, gridspec_kw={"width_ratios": [1, 1.25]})
        for (image_ax, bar_ax), row in zip(axes, page.itertuples()):
            path = (directory / row.id).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Image ID escapes validation directory")
            with Image.open(path) as source:
                image_ax.imshow(source.convert("RGB"), interpolation="nearest")
            image_ax.axis("off")
            image_ax.set_title(f"{row.id} | {'CORRECT' if row.correct else 'ERROR'}\nTrue: {row.true_class}; predicted: {row.predicted_class}", fontsize=10)
            colors = ["#238b45" if name == row.true_class else "#d95f02" if j == 0 else "#6c8ebf"
                      for j, name in enumerate(row.top_classes)]
            bars = bar_ax.barh(row.top_classes, row.top_probabilities, color=colors)
            bar_ax.invert_yaxis()
            bar_ax.set_xlim(0, 1.16)
            bar_ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
            bar_ax.bar_label(bars, labels=[probability_text(v) for v in row.top_probabilities], padding=4)
            bar_ax.set_title(f"Top1−top2: {row.margin:.4f} ({row.margin * 100:.2f} pp)\nTrue probability: {probability_text(row.true_probability)}; rank: {row.true_rank}", fontsize=10)
        fig.suptitle(f"{title}: {start + 1}–{start + len(page)} / {len(cases)}")
        fig.tight_layout(rect=(0, 0, 1, .975), h_pad=2)
        plt.show()
        figures.append(fig)
    return figures


def load_predictions(project_root, experiment_ids):
    """Align every experiment to raw val IDs and verify labels before comparison."""
    root = Path(project_root)
    classes = pd.read_csv(raw_data_dir(root) / 'classes.csv')
    labels = classes.label.tolist()
    if classes.label.duplicated().any() or classes.class_name.duplicated().any():
        raise ValueError("Class labels and names must be unique")
    frame = pd.read_csv(raw_data_dir(root) / 'val.csv', dtype={"id": str})
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
