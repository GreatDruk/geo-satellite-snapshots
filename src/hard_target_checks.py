"""Preflight for paired training: synthetic views/gradients and optional source weights."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import read_classes
from src.experiment import load_config, make_optimizer
from src.hard_target_shift import HardTargetViews, PROFILES, profile_id
from src.hard_target_training import paired_loss, train_paired_epoch, validate_recipe, build_shift_model
from src.models import build_model
from src.transforms import ImageTransform
from src.utils import get_device, set_seed


def check_hard_target_shift(config, project_root=None, device='cpu'):
    cfg = load_config(config)
    validate_recipe(cfg)
    root = Path(project_root or Path(__file__).resolve().parents[1])
    set_seed(cfg['seed'], True, True)
    device = get_device(device)
    # Structured diagnostic image (not test or val) for all transforms.
    yy, xx = np.mgrid[:256, :256]
    array = np.stack([(xx % 32 < 16)*220+20, (yy % 40 < 20)*180+30,
                      ((xx+yy) % 50 < 25)*200+25], axis=2).astype(np.uint8)
    image = Image.fromarray(array)
    views = HardTargetViews(cfg['data'], cfg['seed'])
    traces = []
    for index, name in enumerate(PROFILES):
        clean, hard, trace = views.images(image, 7, 2, forced_profile=index)
        clean_again, hard_again, _ = views.images(image, 7, 2, forced_profile=index)
        assert clean.tobytes() == clean_again.tobytes() and hard.tobytes() == hard_again.tobytes()
        assert clean.size == hard.size == (256,256)
        assert hard.tobytes() != clean.tobytes()
        difference = float(np.abs(np.asarray(clean, dtype=float)-np.asarray(hard, dtype=float)).mean()/255)
        assert difference > .025, f'Profile unexpectedly weak on structured diagnostic image: {name}'
        trace['diagnostic_mean_abs_difference'] = difference
        if name == 'off_nadir':
            corners = np.asarray(trace['corners'])
            assert (corners >= 0).all() and (corners <= 255).all(), 'Entire source frame must fit'
            assert np.asarray(hard).min() > 0, 'No black geometry fill'
        traces.append(trace)
    assert len({trace['d4'] for trace in traces}) == 1
    for index in range(50):
        assert sorted(profile_id(index, epoch) for epoch in range(1,7)) == list(range(6))
    # Stable JS: equal views ->0; disparate logits finite, gradients in both branches.
    first = torch.randn(16,20, requires_grad=True)
    second = torch.randn(16,20, requires_grad=True)
    targets = torch.arange(16) % 20
    loss, terms = paired_loss(first, second, targets)
    terms['js_loss'].backward(retain_graph=True)
    assert first.grad.abs().sum() > 0 and second.grad.abs().sum() > 0
    assert torch.isfinite(loss) and torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()
    _, same = paired_loss(first, first, targets)
    torch.testing.assert_close(same['js_loss'], torch.zeros_like(same['js_loss']), atol=1e-7, rtol=0)
    transform = ImageTransform(cfg['data'], False)
    torch.testing.assert_close(transform(image), transform(image), rtol=0, atol=0)
    assert not transform.augmentation and transform.robust_augmix is None

    # Actual architecture, partial policies and JS+CE backward, no weight download.
    model = build_model(cfg['model'], 256,20, initialize_pretrained=False).to(device)
    optimizer = make_optimizer(model, cfg['training']['optimizer'])
    model.set_epoch(cfg['model']['warmup_epochs']+1)
    model.train()
    prefixes = ('fc.', *[name+'.' for name in cfg['model']['unfreeze_blocks']])
    assert all(parameter.requires_grad == name.startswith(prefixes) for name,parameter in model.backbone.named_parameters())
    bn = [module for module in model.backbone.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    assert all(not module.training for module in bn)
    bn_state = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in bn]
    clean, hard, _ = views(image, 0, 1)
    logits = model(torch.stack((clean,hard)).to(device))
    total, _ = paired_loss(logits[:1], logits[1:], torch.tensor([0], device=device))
    total.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    for module, state in zip(bn, bn_state):
        for actual, saved in zip((module.running_mean, module.running_var, module.num_batches_tracked), state):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    del model, optimizer, logits, total

    class CountForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(3,20)
            self.seen = []
        def forward(self, images):
            self.seen.append(len(images))
            return self.head(images.mean((2,3)))
    small = CountForward()
    loader = DataLoader(TensorDataset(clean.expand(16,-1,-1,-1), hard.expand(16,-1,-1,-1), targets,
                                      torch.arange(16)%6, torch.arange(16)), batch_size=16)
    _, counts = train_paired_epoch(small,loader,torch.optim.AdamW(small.parameters()),torch.device('cpu'))
    assert small.seen == [32] and counts.sum() == 16
    initialization = {'status': 'not_requested'}
    if cfg['training'].get('initialize_from'):
        path = Path(cfg['training']['initialize_from'])
        path = path if path.is_absolute() else root/path
        if path.is_file():
            raw = Path(cfg['data']['raw_dir'])
            raw = raw if raw.is_absolute() else root/raw
            labels = read_classes(raw).label.tolist()
            model, metadata = build_shift_model(cfg, labels, root, device)
            optimizer = make_optimizer(model, cfg['training']['optimizer'])
            assert not optimizer.state
            initialization = {'status': 'verified', **metadata, 'new_optimizer_empty': True}
            del model, optimizer
        else:
            initialization = {'status': 'missing_checkpoint', 'path': str(path),
                              'message': 'Run C requires corrected AGMFA best.pt before GPU training'}
    return {'architecture': cfg['model']['name'], 'device': str(device), 'profiles': traces,
            'coverage_six_epochs': True, 'same_d4': True, 'repeatable_views': True,
            'js_gradients_both_views': True, 'actual_model_forward_backward': True,
            'partial_and_frozen_bn': True, 'original_objects':16, 'forward_images':32,
            'validation_deterministic': True, 'initialization': initialization, 'test_read':False}
