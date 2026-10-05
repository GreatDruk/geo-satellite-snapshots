"""Train objectives; evaluation always uses ordinary, label-free CE logits."""
import math

from torch import nn
from torch.nn import functional as F


def finite_number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


class CEWithCosFace(nn.Module):
    """Ordinary CE plus an auxiliary additive cosine-margin objective.

    The margin applies to true classes during training only. The main CE head
    is the sole inference output; there is no label-conditioned val forward.
    """
    def __init__(self, label_smoothing=0.0, margin=0.15, scale=16.0, weight=0.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.margin = finite_number(margin, "loss.margin", 0, 0.9)
        self.scale = finite_number(scale, "loss.scale", 0.001, 128)
        self.weight = finite_number(weight, "loss.weight", 0.001, 10)

    def forward(self, logits, cosine_scores, labels):
        if cosine_scores.shape != logits.shape:
            raise ValueError("Auxiliary cosine scores must match main logits")
        margin_scores = cosine_scores - self.margin * F.one_hot(labels, logits.size(1)).to(cosine_scores.dtype)
        return self.ce(logits, labels) + self.weight * F.cross_entropy(self.scale * margin_scores, labels)


def make_training_loss(config=None):
    options = dict(config or {})
    name = options.pop("name", "cross_entropy")
    smoothing = finite_number(options.pop("label_smoothing", 0.0), "loss.label_smoothing", 0, 0.99)
    if name == "cross_entropy":
        if options:
            raise ValueError(f"Unsupported cross_entropy settings: {sorted(options)}")
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    if name == "ce_cosface":
        if set(options) - {"margin", "scale", "weight"}:
            raise ValueError(f"Unsupported ce_cosface settings: {sorted(options)}")
        return CEWithCosFace(label_smoothing=smoothing, **options)
    raise ValueError("loss.name must be cross_entropy or ce_cosface")
