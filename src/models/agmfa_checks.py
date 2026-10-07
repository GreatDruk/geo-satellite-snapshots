"""Synthetic preflight on actual torchvision ResNet50; no data or downloads."""
import time

import torch
from torch import nn

from src.experiment import load_config, make_optimizer
from src.models import build_model
from src.transforms import get_transforms
from src.utils import get_device, set_seed


def check_agmfa(config, device="cpu", batch_size=2):
    """Check geometry, masks, gradients, warmup and partial policies in two steps."""
    config = load_config(config)
    assert config["model"]["name"] == "resnet50_agmfa"
    assert config["model"]["mode"] == "partial"
    assert set(config["model"]["unfreeze_blocks"]) == {"layer3", "layer4"}
    assert config["model"]["agmfa"]["attention_source"] == "conv2"
    device = get_device(device)
    set_seed(config["seed"], True, config["training"].get("deterministic_warn_only", True))
    started = time.perf_counter()
    model = build_model(config["model"], 256, 20, initialize_pretrained=False).to(device)
    optimizer = make_optimizer(model, config["training"]["optimizer"])
    images = torch.randn(batch_size, 3, 256, 256, device=device)
    targets = torch.arange(batch_size, device=device) % 20
    bn = [module for module in model.backbone.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    buffers = [(module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone()) for module in bn]
    params = dict(model.backbone.named_parameters())
    early = model.backbone.conv1.weight.detach().clone()
    group_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(group_ids) == len(set(group_ids))
    assert {id(p) for p in model.backbone.fc.parameters()} == {
        id(p) for group in optimizer.param_groups if group["name"] == "head" for p in group["params"]}
    first = model.backbone.layer4[0]
    assert first.conv2.stride == first.downsample[0].stride == (1, 1)
    # Attention is bypassed: compare every manual residual output against the
    # ordinary forward of those SAME blocks, including the residual projection.
    model.eval()
    with torch.inference_mode():
        res2, res3, internals, final = model.multilayer_features(images)
        reference = res3
        equivalence_errors = []
        for index, block in enumerate(model.backbone.layer4):
            captured = []
            handle = block.conv3.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
            try:
                ordinary = block(reference)
            finally:
                handle.remove()
            internal, manual = model.forward_bottleneck(block, reference)
            torch.testing.assert_close(internal, captured[0], rtol=0, atol=0)
            torch.testing.assert_close(internal, internals[index], rtol=0, atol=0)
            torch.testing.assert_close(manual, ordinary, rtol=0, atol=0)
            equivalence_errors.append((manual - ordinary).abs().max().item())
            reference = ordinary
        torch.testing.assert_close(final, reference, rtol=0, atol=0)
    del res2, res3, internals, final, reference, ordinary, internal, manual, captured
    shapes = None
    for epoch in (1, config["model"]["warmup_epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        model.set_epoch(epoch)
        model.train()
        prefixes = ("fc.",) if epoch == 1 else ("fc.", "layer3.", "layer4.")
        assert all(p.requires_grad == name.startswith(prefixes) for name, p in params.items())
        assert all(not module.training for module in bn)
        assert model.backbone.fc.fusion[-1].training, "New GroupNorm must stay trainable"
        details = model.forward_details(images)
        expected = {"res2": (512, 32, 32), "res3": (1024, 16, 16), "res4": (2048, 16, 16),
                    **{f"res4_{i}": (512, 16, 16) for i in (1, 2, 3)}}
        for name, shape in expected.items():
            assert details[name].shape == (batch_size, *shape), name
        for i in (1, 2, 3):
            for kind, shape in (("spatial", (1, 16, 16)), ("channel", (512, 1, 1))):
                mask = details[f"{kind}_mask_{i}"]
                assert mask.shape == (batch_size, *shape)
                assert torch.isfinite(mask).all() and (mask >= 0).all() and (mask <= 1).all()
        torch.testing.assert_close(details["semantic_mask"],
                                   details["spatial_mask_1"] * details["spatial_mask_2"] * details["spatial_mask_3"])
        for i in (2, 3, 4):
            torch.testing.assert_close(details[f"guided_res{i}"], details[f"aligned_res{i}"] * details["semantic_mask"])
        assert details["concatenated"].shape == (batch_size, 3584, 16, 16)
        torch.testing.assert_close(details["aligned_res4"], details["res4"], rtol=0, atol=0)
        assert details["logits"].shape == (batch_size, 20)
        loss = nn.CrossEntropyLoss()(details["logits"], targets)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in params.items():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            else:
                assert parameter.grad is None, name
        for attention in model.backbone.fc.attentions:
            assert attention.spatial.weight.grad.abs().sum() > 0
            assert attention.mlp[0].weight.grad.abs().sum() > 0
        changed = model.backbone.fc.classifier[-1].weight if epoch == 1 else first.conv2.weight
        before = changed.detach().clone()
        optimizer.step()
        assert not torch.equal(before, changed)
        torch.testing.assert_close(early, model.backbone.conv1.weight, rtol=0, atol=0)
        if shapes is None:
            shapes = {name: list(value.shape) for name, value in details.items()}
        del details, loss
    for module, expected in zip(bn, buffers):
        for actual, original in zip((module.running_mean, module.running_var, module.num_batches_tracked), expected):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert group_ids == [id(p) for group in optimizer.param_groups for p in group["params"]]
    train_transform, val_transform = get_transforms(config["data"])
    assert train_transform.robust_augmix is not None
    assert val_transform.robust_augmix is None and not val_transform.augmentation
    model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(images), model(images), rtol=0, atol=0)
    return {"device": str(device), "shapes": shapes, "masks_product_verified": True,
            "manual_bottlenecks_match_ordinary": True, "bottleneck_max_abs_errors": equivalence_errors,
            "all_attention_gradients": True, "warmup_and_partial_updates": True,
            "early_backbone_unchanged": True, "backbone_bn_buffers_unchanged": True,
            "optimizer_groups_stable": True, "validation_deterministic": True,
            "pretrained_downloaded": False, "test_read": False,
            "parameters": sum(p.numel() for p in model.parameters()),
            "optimizer_lrs": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "elapsed_seconds": time.perf_counter() - started}
