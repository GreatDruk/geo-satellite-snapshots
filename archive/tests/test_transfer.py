"""Transfer policies on a tiny backbone; optional real torchvision/CUDA checks."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from archive.src.experiment import make_optimizer, load_checkpoint, load_config, PROJECT_ROOT
from archive.src.models import build_model
from archive.src.models.pooling import CompatibleAdaptiveAvgPool2d
from archive.src.training.train import train_model
from archive.src.utils import set_seed


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        for index in range(1, 5):
            setattr(self, f"layer{index}", nn.Sequential(nn.Conv2d(4, 4, 3, padding=1),
                                                       nn.BatchNorm2d(4), nn.ReLU(), nn.Dropout(0.1)))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1000)

    def forward(self, images):
        output = self.bn1(self.conv1(images))
        for index in range(1, 5):
            output = getattr(self, f"layer{index}")(output)
        return self.fc(self.avgpool(output).flatten(1))


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def build(self, mode, **options):
        config = {"name": "resnet18", "pretrained": True, "weights": "IMAGENET1K_V1",
                  "mode": mode, "head_dropout": 0.2, **options}
        with patch("archive.src.models.pretrained.load_backbone", return_value=(TinyResNet(), "IMAGENET1K_V1")):
            return build_model(config, 16, 2), config

    def step(self, model):
        optimizer = make_optimizer(model, {"name": "adamw", "lr": 0.001,
                                            "head_lr": 0.001, "backbone_lr": 0.0001})
        train_model(model, [(torch.randn(3, 3, 16, 16), torch.tensor([0, 1, 0]))],
                    nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))
        return optimizer

    def test_head_only_freezes_weights_bn_and_dropout(self):
        model, _ = self.build("head_only")
        before = {name: value.clone() for name, value in model.backbone.state_dict().items()}
        optimizer = self.step(model)
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["head"])
        for name, tensor in model.backbone.state_dict().items():
            if not name.startswith("fc."):
                torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        self.assertFalse(model.backbone.bn1.training)
        self.assertFalse(model.backbone.layer3[-1].training)
        self.assertTrue(model.backbone.fc.training)
        self.assertFalse(torch.equal(before["fc.1.weight"], model.backbone.fc[1].weight))

    def test_partial_only_changes_selected_block_and_head(self):
        model, _ = self.build("partial", unfreeze_blocks=["layer4"])
        before = {name: value.clone() for name, value in model.backbone.state_dict().items()}
        optimizer = self.step(model)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.0001, 0.001])
        for name, tensor in model.backbone.state_dict().items():
            if not name.startswith(("layer4.", "fc.")):
                torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        self.assertTrue(model.backbone.layer4[1].training)
        self.assertFalse(torch.equal(before["layer4.0.weight"], model.backbone.layer4[0].weight))

    def test_full_and_warmup_keep_optimizer_groups_stable(self):
        model, _ = self.build("full", warmup_epochs=2, freeze_bn=True)
        model.set_epoch(1)
        optimizer = self.step(model)
        self.assertIsNone(model.backbone.conv1.weight.grad)
        self.assertFalse(model.backbone.bn1.training)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        model.set_epoch(3)
        model.train()
        logits = model(torch.randn(2, 3, 16, 16))
        logits.sum().backward()
        self.assertIsNotNone(model.backbone.conv1.weight.grad)
        self.assertEqual(ids, [id(p) for group in optimizer.param_groups for p in group["params"]])

    def test_checkpoint_restore_never_downloads_initial_weights(self):
        model, model_config = self.build("partial", unfreeze_blocks=["layer4"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"config": {"model": model_config, "data": {"image_size": 16}},
                        "class_labels": [4, 11], "epoch": 5,
                        "model_state_dict": model.state_dict()}, path)
            with patch("archive.src.models.pretrained.load_backbone", return_value=(TinyResNet(), "IMAGENET1K_V1")) as loader:
                restored, _ = load_checkpoint(path, "cpu")
                self.assertFalse(loader.call_args.args[2])
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)

    def test_pool_output_and_gradient_match_adaptive_definition(self):
        for shape, output in (((2, 3, 16, 16), (4, 4)), ((2, 3, 15, 17), (1, 1)),
                              ((2, 3, 15, 17), (4, 3))):
            original = torch.randn(shape, requires_grad=True)
            replacement = original.detach().clone().requires_grad_(True)
            expected = nn.AdaptiveAvgPool2d(output)(original)
            actual = CompatibleAdaptiveAvgPool2d(output)(replacement)
            torch.testing.assert_close(actual, expected)
            upstream = torch.randn_like(expected)
            expected.backward(upstream)
            actual.backward(upstream)
            torch.testing.assert_close(replacement.grad, original.grad)

    def test_seed_strict_and_warning_modes(self):
        set_seed(42, True, True)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        set_seed(42, True, False)
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_invalid_partial_block(self):
        with self.assertRaises(ValueError):
            self.build("partial", unfreeze_blocks=["typo"])

    @unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision not installed")
    def test_real_resnet18_without_downloading(self):
        model = build_model({"name": "resnet18", "pretrained": False, "mode": "partial",
                             "unfreeze_blocks": ["layer4"]}, 64, 20)
        model.train()
        logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(logits.shape, (2, 20))
        logits.sum().backward()
        self.assertIsNone(model.backbone.layer1[0].conv1.weight.grad)
        self.assertIsNotNone(model.backbone.layer4[0].conv1.weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_cuda_custom_cnn_strict_backward(self):
        set_seed(42, True, False)
        for name in ("cnn_256_balanced", "cnn_256_deeper"):
            config = load_config(PROJECT_ROOT / f"configs/{name}.json")
            model = build_model(config["model"], 256, 20).cuda()
            logits = model(torch.randn(2, 3, 256, 256, device="cuda"))
            logits.sum().backward()


if __name__ == "__main__":
    unittest.main()
