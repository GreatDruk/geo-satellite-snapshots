"""Synthetic-only preflight for the real torchvision ResNet50 + LGFFE."""
import time

import torch
from torch import nn

from src.experiment import load_config, make_optimizer
from src.models import build_model
from src.utils import get_device, set_seed


def check_lgffe(config, device="cpu", batch_size=2):
    """Two synthetic optimizer steps, no dataset access or pretrained download.

The real ResNet50 geometry and all policies are tested with random initial
backbone tensors. The actual experiment separately loads IMAGENET1K_V2.
"""
    config = load_config(config)
    if config["model"]["name"] != "resnet50_lgffe":
        raise ValueError("Expected a resnet50_lgffe config")
    if config["model"]["mode"] != "partial" or set(config["model"]["unfreeze_blocks"]) != {"layer3", "layer4"}:
        raise ValueError("This preflight checks partial layer3+layer4")
    if batch_size < 2:
        raise ValueError("Use at least two synthetic images")
    device = get_device(device)
    set_seed(config["seed"], True, config["training"].get("deterministic_warn_only", True))
    started = time.perf_counter()
    model = build_model(config["model"], config["data"]["image_size"], 20,
                        initialize_pretrained=False).to(device)
    optimizer = make_optimizer(model, config["training"]["optimizer"])
    size = config["data"]["image_size"]
    size = (size, size) if isinstance(size, int) else tuple(size)
    images = torch.randn(batch_size, 3, *size, device=device)
    targets = torch.arange(batch_size, device=device) % 20
    criterion = nn.CrossEntropyLoss()
    bn_modules = [module for module in model.backbone.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    bn_state = [(module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
                for module in bn_modules]
    params = dict(model.backbone.named_parameters())
    group_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(group_ids) == len(set(group_ids)), "Repeated optimizer parameter"
    head_ids = {id(parameter) for parameter in model.backbone.fc.parameters()}
    assert head_ids == {id(parameter) for group in optimizer.param_groups if group["name"] == "head" for parameter in group["params"]}

    model.set_epoch(1)
    model.train()
    mixstyle_modules = list(model.mixstyle.values()) if hasattr(model, "mixstyle") else []
    assert all(module.training for module in mixstyle_modules), "MixStyle must stay active during warmup"
    assert all(parameter.requires_grad == name.startswith("fc.") for name, parameter in params.items())
    assert all(not module.training for module in bn_modules)
    before_head = model.backbone.fc.classifier[-1].weight.detach().clone()
    before_early = model.backbone.conv1.weight.detach().clone()
    details = model.forward_details(images)
    attention = details["attention"]
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(batch_size, device=device))
    assert (attention >= 0).all() and torch.isfinite(attention).all()
    torch.testing.assert_close(details["global_features"], details["feature_map"].mean((2, 3)))
    torch.testing.assert_close(details["local_features"], (details["regions"] * attention.unsqueeze(-1)).sum(1))
    assert details["logits"].shape == (batch_size, 20)
    loss = criterion(details["logits"], targets)
    loss.backward()
    for name, parameter in params.items():
        if name.startswith("fc."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    assert model.backbone.fc.gru.weight_ih_l0.grad.abs().sum() > 0
    assert model.backbone.fc.region_refinement.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before_head, model.backbone.fc.classifier[-1].weight)
    torch.testing.assert_close(before_early, model.backbone.conv1.weight, rtol=0, atol=0)
    shapes = {name: list(value.shape) for name, value in details.items()}
    del details, attention, loss

    # The existing set_epoch hook unlocks eligible body tensors after warmup.
    optimizer.zero_grad(set_to_none=True)
    model.set_epoch(config["model"]["warmup_epochs"] + 1)
    model.train()
    assert all(module.training for module in mixstyle_modules), "MixStyle must stay active during partial fine-tuning"
    for name, parameter in params.items():
        assert parameter.requires_grad == name.startswith(("fc.", "layer3.", "layer4.")), name
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    for name, parameter in params.items():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    before_late = model.backbone.layer4[0].conv1.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before_late, model.backbone.layer4[0].conv1.weight)
    torch.testing.assert_close(before_early, model.backbone.conv1.weight, rtol=0, atol=0)
    assert group_ids == [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    for module, (mean, variance, count) in zip(bn_modules, bn_state):
        assert not module.training
        for actual, expected in ((module.running_mean, mean), (module.running_var, variance),
                                 (module.num_batches_tracked, count)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model.eval()
    for module in mixstyle_modules:
        assert not module.training
        example = torch.randn(2, 4, 8, 8, device=device)
        state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        assert module(example) is example, "MixStyle eval must be an exact no-op"
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        if cuda_state is not None:
            torch.testing.assert_close(torch.cuda.get_rng_state(device), cuda_state, rtol=0, atol=0)
    return {"device": str(device), "torch": str(torch.__version__), "shapes": shapes,
            "attention_normalized": True, "gru_and_attention_gradients": True,
            "warmup_head_updated": True, "partial_layer3_layer4_updated": True,
            "early_backbone_unchanged": True, "bn_buffers_unchanged": True,
            "optimizer_groups_stable": True, "pretrained_downloaded": False,
            "mixstyle_modules": len(mixstyle_modules), "mixstyle_train_active_eval_noop": bool(mixstyle_modules),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "optimizer_lrs": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "elapsed_seconds": time.perf_counter() - started}
