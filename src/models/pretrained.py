"""Torchvision ResNets with explicit transfer-learning and BatchNorm policies."""
from torch import nn
from torch.nn import functional as F

from .pooling import replace_adaptive_pools, GlobalAverageMaxPool2d

RESNET_ARCHITECTURES = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}


def load_backbone(architecture, weights_name, download_weights):
    try:
        from torchvision import models
    except (ImportError, RuntimeError) as error:
        raise ImportError("Pretrained ResNets require a torchvision build compatible with torch") from error
    weights = None
    if download_weights:
        enum = models.get_model_weights(architecture)
        try:
            weights = enum[weights_name]
        except KeyError as error:
            raise ValueError(f"Unknown weights {weights_name} for {architecture}") from error
    backbone = models.get_model(architecture, weights=weights)
    return backbone, weights.name if weights is not None else weights_name


class TransferClassifier(nn.Module):
    """Frozen subtrees remain in eval mode even after the epoch calls model.train()."""
    def __init__(self, backbone, config, architecture):
        super().__init__()
        self.backbone = backbone
        auxiliary = config.get("auxiliary_cosine_head", False)
        if not isinstance(auxiliary, bool):
            raise ValueError("auxiliary_cosine_head must be boolean")
        if auxiliary:
            self.auxiliary_head = nn.Linear(backbone.fc[1].in_features, backbone.fc[1].out_features, bias=False)
        self.architecture = architecture
        self.mode = config.get("mode", "full")
        self.freeze_bn = config.get("freeze_bn", False)
        self.warmup_epochs = config.get("warmup_epochs", 0)
        if self.mode not in {"head_only", "partial", "full"}:
            raise ValueError("Transfer mode must be head_only, partial or full")
        if not isinstance(self.warmup_epochs, int) or self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a nonnegative integer")
        blocks = config.get("unfreeze_blocks", [])
        if self.mode == "partial" and not blocks:
            raise ValueError("partial mode requires unfreeze_blocks, e.g. ['layer4']")
        if self.mode != "partial" and blocks:
            raise ValueError("unfreeze_blocks is only used with partial mode")
        for block in blocks:
            if block not in {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"}:
                raise ValueError(f"Unknown ResNet block: {block}")
        self.eligible_names = {
            name for name, _ in backbone.named_parameters()
            if name.startswith("fc.") or self.mode == "full"
            or (self.mode == "partial" and any(name == block or name.startswith(block + ".") for block in blocks))
        }
        self.set_epoch(self.warmup_epochs + 1)

    def set_epoch(self, epoch):
        self.active_mode = "head_only" if epoch <= self.warmup_epochs else self.mode
        for name, parameter in self.backbone.named_parameters():
            enabled = name.startswith("fc.") or (self.active_mode != "head_only" and name in self.eligible_names)
            parameter.requires_grad_(enabled)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                parameters = list(module.parameters())
                if parameters and not any(parameter.requires_grad for parameter in parameters):
                    module.eval()
                if self.freeze_bn and isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def optimizer_groups(self, base_lr, backbone_lr=None, head_lr=None):
        head, body = [], []
        for name, parameter in self.backbone.named_parameters():
            if name in self.eligible_names:
                (head if name.startswith("fc.") else body).append(parameter)
        groups = []
        if body:
            groups.append({"params": body, "lr": base_lr if backbone_lr is None else backbone_lr, "name": "backbone"})
        if hasattr(self, "auxiliary_head"):
            head.extend(self.auxiliary_head.parameters())
        groups.append({"params": head, "lr": base_lr if head_lr is None else head_lr, "name": "head"})
        return groups

    def forward(self, images):
        return self.backbone(images)

    def forward_with_auxiliary(self, images):
        """One ResNet pass, ordinary logits plus normalized auxiliary similarities."""
        if not hasattr(self, "auxiliary_head"):
            raise ValueError("ce_cosface requires model.auxiliary_cosine_head=true")
        body = self.backbone
        features = body.bn1(body.conv1(images))
        features = getattr(body, "relu", nn.Identity())(features)
        features = getattr(body, "maxpool", nn.Identity())(features)
        for block in (body.layer1, body.layer2, body.layer3, body.layer4):
            features = block(features)
        features = body.avgpool(features).flatten(1)
        cosine = F.linear(F.normalize(features, dim=1), F.normalize(self.auxiliary_head.weight, dim=1))
        return body.fc(features), cosine


def build_pretrained(config, num_classes, initialize_pretrained=True):
    architecture = config.get("architecture", config["name"]).lower()
    if architecture == "resnet":
        architecture = "resnet18"
    if architecture not in RESNET_ARCHITECTURES:
        raise ValueError(f"Built-in transfer models: {sorted(RESNET_ARCHITECTURES)}")
    weights_name = config.get("weights", "DEFAULT")
    if config.get("pretrained", False) and not isinstance(weights_name, str):
        raise ValueError("Pretrained weights must be a named torchvision weight set")
    backbone, resolved_weights = load_backbone(architecture, weights_name,
                                               initialize_pretrained and config.get("pretrained", False))
    pooling = config.get("pooling", "average")
    if pooling not in {"average", "avgmax"}:
        raise ValueError("ResNet pooling must be average or avgmax")
    feature_count = backbone.fc.in_features
    if pooling == "avgmax":
        backbone.avgpool = GlobalAverageMaxPool2d()
        feature_count *= 2
    backbone.fc = nn.Sequential(nn.Dropout(config.get("head_dropout", 0.3)),
                                nn.Linear(feature_count, num_classes))
    replace_adaptive_pools(backbone)
    model = TransferClassifier(backbone, config, architecture)
    model.weights_name = resolved_weights
    return model
