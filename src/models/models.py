"""Config-driven custom MLP/CNN producing raw logits."""
from math import prod
from torch import nn


def build_model(config, image_size, num_classes, model_factory=None):
    """External factory: (model_config, image_size, num_classes) -> nn.Module."""
    if config.get("pretrained", False) and config["name"].lower() not in {"alexnet", "vgg", "resnet", "inception"}:
        raise ValueError("Only pretrained AlexNet, VGG, ResNet and Inception are permitted")
    if model_factory is not None:
        model = model_factory(config, image_size, num_classes)
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return nn.Module producing class logits")
        return model
    if config.get("pretrained", False) or config["name"] not in {"mlp", "cnn"}:
        raise ValueError("Built-in models: mlp/cnn. Others require model_factory")
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    shape, layers = (3, *size), []
    constructors = {
        "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid, "leaky_relu": nn.LeakyReLU, "dropout": nn.Dropout,
    }
    def pair(value):
        return (value, value) if isinstance(value, int) else tuple(value)

    for specification in config["layers"]:
        options = dict(specification)
        kind = options.pop("type")
        if kind == "flatten":
            if options:
                raise ValueError("flatten has no configurable options")
            layer, shape = nn.Flatten(), (prod(shape),)
        elif kind == "linear":
            if len(shape) != 1:
                raise ValueError("Add flatten before linear")
            width = options.pop("out_features")
            width = num_classes if width == "num_classes" else width
            layer, shape = nn.Linear(shape[0], width, **options), (width,)
        elif kind == "conv2d":
            if len(shape) != 3:
                raise ValueError("conv2d needs spatial input")
            layer = nn.Conv2d(shape[0], **options)
            spatial = tuple((shape[i + 1] + 2 * layer.padding[i] - layer.dilation[i] * (layer.kernel_size[i] - 1) - 1) // layer.stride[i] + 1 for i in range(2))
            shape = (layer.out_channels, *spatial)
        elif kind == "max_pool2d":
            if len(shape) != 3 or options.get("ceil_mode", False):
                raise ValueError("max_pool2d needs spatial input and ceil_mode=false")
            layer = nn.MaxPool2d(**options)
            kernel, stride = pair(layer.kernel_size), pair(layer.stride)
            padding, dilation = pair(layer.padding), pair(layer.dilation)
            shape = (shape[0], *((shape[i + 1] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1 for i in range(2)))
        elif kind == "adaptive_avg_pool2d":
            if len(shape) != 3:
                raise ValueError("adaptive_avg_pool2d needs spatial input")
            layer = nn.AdaptiveAvgPool2d(**options)
            output = pair(layer.output_size)
            shape = (shape[0], *(shape[i + 1] if output[i] is None else output[i] for i in range(2)))
        elif kind == "batch_norm2d":
            if len(shape) != 3:
                raise ValueError("batch_norm2d needs spatial input")
            layer = nn.BatchNorm2d(shape[0], **options)
        elif kind in constructors:
            layer = constructors[kind](**options)
        else:
            raise ValueError(f"Unsupported layer: {kind}")
        if any(value <= 0 for value in shape):
            raise ValueError(f"Invalid output shape after {kind}: {shape}")
        layers.append(layer)
    if shape != (num_classes,) or config["layers"][-1]["type"] != "linear":
        raise ValueError("End with a linear layer producing num_classes raw logits")
    return nn.Sequential(*layers)
