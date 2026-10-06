"""Local-global fusion with GRU-based regional attention (Lv et al., 2019).

Equations 6-9: shared GRU-state scoring, region-to-region score refinement,
spatial softmax, then a weighted sum of the ORIGINAL regional CNN features.
Dimensions adapt to ResNet50 layer4; no crop, QKV or Transformer is involved.
"""
import torch
from torch import nn

from .pretrained import TransferClassifier, load_backbone
from .mixstyle import MixStyle


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class LGFFEHead(nn.Module):
    """Receive [B,C,H,W], return logits; inspect attention via forward_details."""

    def __init__(self, channels, spatial_size, num_classes, *, gru_hidden_size=512,
                 head_hidden_size=512, dropout=0.3):
        super().__init__()
        self.channels = channels
        self.spatial_size = tuple(spatial_size)
        self.num_regions = self.spatial_size[0] * self.spatial_size[1]
        hidden = _positive_int(gru_hidden_size, "gru_hidden_size")
        head_hidden = _positive_int(head_hidden_size, "head_hidden_size")
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.region_score = nn.Linear(hidden, 1)
        # Eq. 7 mixes the scalar scores of all regions, not channel features.
        self.region_refinement = nn.Linear(self.num_regions, self.num_regions)
        self.classifier = nn.Sequential(
            nn.Linear(2 * channels, head_hidden), nn.LayerNorm(head_hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, num_classes),
        )

    def forward_details(self, feature_map):
        if feature_map.ndim != 4 or feature_map.shape[1:] != (self.channels, *self.spatial_size):
            raise ValueError(f"LGFFE expects [B,{self.channels},{self.spatial_size[0]},{self.spatial_size[1]}]; "
                             f"received {tuple(feature_map.shape)}. Use the checkpoint image_size.")
        # Row-major scan, width varying fastest; GRU models regional context.
        regions = feature_map.flatten(2).transpose(1, 2).contiguous()
        contextual_regions, _ = self.gru(regions)
        scores = torch.relu(self.region_score(contextual_regions).squeeze(-1))
        attention = self.region_refinement(scores).softmax(dim=1)
        global_features = feature_map.mean(dim=(2, 3))
        local_features = (regions * attention.unsqueeze(-1)).sum(dim=1)
        fused = torch.cat((global_features, local_features), dim=1)
        return {
            "logits": self.classifier(fused), "feature_map": feature_map,
            "regions": regions, "contextual_regions": contextual_regions,
            "attention": attention, "global_features": global_features,
            "local_features": local_features, "fused_features": fused,
        }

    def forward(self, feature_map):
        return self.forward_details(feature_map)["logits"]


class LGFFEClassifier(TransferClassifier):
    """Reuse ResNet transfer policies; the entire extractor lives under fc.*.

Thus the GRU, attention and classifier are all in the head LR group and remain
trainable during head-only warmup. Ordinary ResNet state keys are unchanged.
"""

    def __init__(self, backbone, config, architecture):
        super().__init__(backbone, config, architecture)
        settings = config.get("mixstyle")
        if settings is not None:
            if not isinstance(settings, dict) or set(settings) - {"p", "alpha", "eps", "after_blocks"}:
                raise ValueError("MixStyle settings: p, alpha, eps, after_blocks")
            blocks = settings.get("after_blocks", ["layer1", "layer2"])
            if not isinstance(blocks, list) or blocks != ["layer1", "layer2"]:
                raise ValueError("LGFFE MixStyle must follow layer1 and layer2")
            options = {key: value for key, value in settings.items() if key != "after_blocks"}
            # Outside frozen backbone subtrees: TransferClassifier.train()
            # may set layer1/layer2 to eval but MixStyle must stay in train.
            self.mixstyle = nn.ModuleDict({block: MixStyle(**options) for block in blocks})

    def feature_map(self, images):
        body = self.backbone
        features = body.maxpool(body.relu(body.bn1(body.conv1(images))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            features = getattr(body, name)(features)
            if hasattr(self, "mixstyle") and name in self.mixstyle:
                features = self.mixstyle[name](features)
        return features

    def forward(self, images):
        # Bypass ResNet's standard avgpool and fc invocation on flattened data.
        return self.backbone.fc(self.feature_map(images))

    def forward_details(self, images):
        return self.backbone.fc.forward_details(self.feature_map(images))

    def forward_with_auxiliary(self, images):
        raise ValueError("LGFFE supports ordinary CrossEntropyLoss, not the auxiliary CosFace head")


def build_lgffe(config, image_size, num_classes, initialize_pretrained=True):
    if config.get("auxiliary_cosine_head", False) or config.get("pooling", "average") != "average":
        raise ValueError("LGFFE requires GAP plus GRU attention and no auxiliary cosine head")
    settings = dict(config.get("lgffe", {}))
    if set(settings) - {"gru_hidden_size", "head_hidden_size"}:
        raise ValueError(f"Unknown LGFFE settings: {sorted(settings)}")
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    if len(size) != 2:
        raise ValueError("LGFFE image_size must be an integer or [height, width]")
    for length in size:
        _positive_int(length, "image_size")
    # Torchvision ResNet50 has total stride 32, with padded convolutions.
    spatial = tuple((length + 31) // 32 for length in size)
    weights_name = config.get("weights", "IMAGENET1K_V2")
    if not isinstance(weights_name, str):
        raise ValueError("Pretrained weights must be a named torchvision weight set")
    backbone, resolved_weights = load_backbone(
        "resnet50", weights_name, initialize_pretrained and config.get("pretrained", False),
    )
    channels = backbone.fc.in_features
    backbone.fc = LGFFEHead(channels, spatial, num_classes,
                            dropout=config.get("head_dropout", 0.3), **settings)
    model = LGFFEClassifier(backbone, config, "resnet50_lgffe")
    model.weights_name = resolved_weights
    return model
