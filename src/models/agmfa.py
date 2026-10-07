"""AGMFA: high-resolution ResNet, three semantic masks, multilayer fusion.

Li et al., Remote Sensing 13(16), 3113 (2021), sections 3.1--3.3.
Attention uses internal conv2/bn2/ReLU features; fusion uses full stage outputs.
GroupNorm is an explicit normalization choice for our frozen-BN transfer recipe.
"""
import torch
from torch import nn

from .pretrained import TransferClassifier, load_backbone


class ChannelSpatialMask(nn.Module):
    """CBAM ordering: shared channel MLP, then spatial sigmoid mask."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(),
                                 nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, features):
        channel = (self.mlp(features.mean((2, 3), keepdim=True))
                   + self.mlp(features.amax((2, 3), keepdim=True))).sigmoid()
        weighted = features * channel
        spatial = self.spatial(torch.cat((weighted.mean(1, keepdim=True),
                                         weighted.amax(1, keepdim=True)), 1)).sigmoid()
        return channel, spatial


class AGMFAHead(nn.Module):
    """Attention-guided Res2/Res3/Res4 concatenation; all new parameters here."""
    def __init__(self, channels, num_classes, *, attention_reduction=16,
                 fusion_channels=512, normalization="group_norm", norm_groups=32,
                 dropout=0.3, attention_channels=None):
        super().__init__()
        for name, value in (("attention_reduction", attention_reduction),
                            ("fusion_channels", fusion_channels), ("norm_groups", norm_groups)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if normalization != "group_norm" or fusion_channels % norm_groups:
            raise ValueError("Use group_norm with norm_groups dividing fusion_channels")
        self.attentions = nn.ModuleList(ChannelSpatialMask(
            channels[-1] if attention_channels is None else attention_channels, attention_reduction)
                                        for _ in range(3))
        self.res2_pool = nn.MaxPool2d(2, 2)
        # Concatenation permits different channel counts. This 1x1 convolution
        # aligns the concatenated 512+1024+2048 channels to fusion_channels.
        self.fusion = nn.Sequential(nn.Conv2d(sum(channels), fusion_channels, 1, bias=False),
                                    nn.ReLU(), nn.GroupNorm(norm_groups, fusion_channels))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_channels, num_classes))

    def forward_details(self, res2, res3, res4_maps, res4):
        if len(res4_maps) != 3:
            raise ValueError("AGMFA requires features from all three layer4 bottlenecks")
        pairs = [attention(features) for attention, features in zip(self.attentions, res4_maps)]
        masks = [pair[1] for pair in pairs]
        mask = masks[0] * masks[1] * masks[2]
        aligned = [self.res2_pool(res2), res3, res4]
        if any(features.shape[-2:] != mask.shape[-2:] for features in aligned):
            raise ValueError("Res2 pooled, Res3 and modified Res4 must have equal spatial sizes")
        guided = [features * mask for features in aligned]
        concatenated = torch.cat(guided, dim=1)
        fused = self.fusion(concatenated)
        pooled = fused.mean((2, 3))
        details = {"res2": res2, "res3": res3, "res4": res4, "semantic_mask": mask,
                   "concatenated": concatenated, "fused_map": fused,
                   "fused_features": pooled, "logits": self.classifier(pooled)}
        for index, (features, pair) in enumerate(zip(res4_maps, pairs), 1):
            details.update({f"res4_{index}": features, f"channel_mask_{index}": pair[0],
                            f"spatial_mask_{index}": pair[1]})
        for index, (features, weighted) in enumerate(zip(aligned, guided), 2):
            details.update({f"aligned_res{index}": features, f"guided_res{index}": weighted})
        return details

    def forward(self, res2, res3, res4_maps, res4):
        return self.forward_details(res2, res3, res4_maps, res4)["logits"]


class AGMFAClassifier(TransferClassifier):
    """Existing warmup, partial freezing and two LR groups remain applicable."""
    def __init__(self, backbone, config, architecture):
        super().__init__(backbone, config, architecture)
        # Missing field occurs only in historical checkpoint configs. Preserve
        # their 2048-channel heads; the corrected recipe explicitly selects conv2.
        self.attention_source = config.get("agmfa", {}).get("attention_source", "block_output")

    @staticmethod
    def forward_bottleneck(block, features):
        """Torchvision Bottleneck forward, also returning its pre-conv3 features."""
        identity = features
        output = block.relu(block.bn1(block.conv1(features)))
        internal = block.relu(block.bn2(block.conv2(output)))
        output = block.bn3(block.conv3(internal))
        if block.downsample is not None:
            identity = block.downsample(features)
        output += identity
        return internal, block.relu(output)

    def multilayer_features(self, images):
        body = self.backbone
        features = body.maxpool(body.relu(body.bn1(body.conv1(images))))
        res2 = body.layer2(body.layer1(features))
        res3 = body.layer3(res2)
        features, res4_maps = res3, []
        for block in body.layer4:
            internal, features = self.forward_bottleneck(block, features)
            res4_maps.append(internal if self.attention_source == "conv2" else features)
        return res2, res3, res4_maps, features

    def forward_details(self, images):
        return self.backbone.fc.forward_details(*self.multilayer_features(images))

    def forward(self, images):
        return self.backbone.fc(*self.multilayer_features(images))

    def forward_with_auxiliary(self, images):
        raise ValueError("AGMFA experiment uses ordinary CrossEntropyLoss")


def build_agmfa(config, image_size, num_classes, initialize_pretrained=True):
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    if size != (256, 256):
        raise ValueError("This AGMFA experiment requires full 256x256 inputs")
    if config.get("mixstyle") is not None or config.get("auxiliary_cosine_head", False):
        raise ValueError("AGMFA experiment excludes MixStyle and auxiliary heads")
    if config.get("pooling", "average") != "average":
        raise ValueError("AGMFA uses global average pooling")
    settings = dict(config.get("agmfa", {}))
    source = settings.pop("attention_source", "block_output")
    if source not in {"conv2", "block_output"}:
        raise ValueError("attention_source must be conv2 (fixed) or block_output (historical)")
    if set(settings) - {"attention_reduction", "fusion_channels", "normalization", "norm_groups"}:
        raise ValueError(f"Unknown AGMFA settings: {sorted(settings)}")
    backbone, weights = load_backbone("resnet50", config.get("weights", "IMAGENET1K_V2"),
                                      initialize_pretrained and config.get("pretrained", False))
    if len(backbone.layer4) != 3:
        raise ValueError("ResNet50 layer4 must have three bottlenecks")
    first = backbone.layer4[0]
    # Torchvision bottleneck puts the spatial stride in conv2, and the
    # residual projection must be changed too. Weight tensors stay compatible.
    first.conv2.stride = (1, 1)
    first.downsample[0].stride = (1, 1)
    first.stride = 1
    channels = tuple(getattr(backbone, name)[-1].conv3.out_channels
                     for name in ("layer2", "layer3", "layer4"))
    attention_channels = first.conv2.out_channels if source == "conv2" else channels[-1]
    backbone.fc = AGMFAHead(channels, num_classes, attention_channels=attention_channels,
                            dropout=config.get("head_dropout", 0.3), **settings)
    model = AGMFAClassifier(backbone, config, "resnet50_agmfa")
    model.weights_name = weights
    return model
