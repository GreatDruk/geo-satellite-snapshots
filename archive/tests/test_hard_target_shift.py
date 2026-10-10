"""Paired views, persistent-worker epoch propagation, JS and artifact integration."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from archive.src.experiment import load_checkpoint, make_optimizer
from archive.src.hard_target_shift import HardTargetViews, HardTargetDataset, PROFILES, profile_id
from archive.src.hard_target_training import run_hard_target_shift, paired_loss, build_shift_model
from archive.src.hard_target_checks import check_hard_target_shift
from archive.src.data import make_loader
from archive.src.models import build_model
from archive.src.training.predict import load_probabilities, probability_predictions
from archive.tests.test_agmfa import TinyResNet

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/resnet50_agmfa_hard_target_shift.json'


def fixture(root, count=6):
    raw = root/'data/raw'
    for split in ('train','val'):
        (raw/split).mkdir(parents=True)
        rows = []
        for index in range(count):
            image_id = f'{index}.png'
            array = np.random.default_rng(index+42).integers(20,240,(256,256,3),dtype=np.uint8)
            Image.fromarray(array).save(raw/split/image_id)
            rows.append({'id':image_id,'label':[3,9][index%2]})
        pd.DataFrame(rows).to_csv(raw/f'{split}.csv',index=False)
    pd.DataFrame({'label':[3,9], 'class_name':['church','palace']}).to_csv(raw/'classes.csv',index=False)
    return raw


class HardShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_profile_cycle_and_shared_d4_repeatability(self):
        cfg = json.loads(CONFIG.read_text())
        views = HardTargetViews(cfg['data'])
        image = Image.fromarray(np.random.default_rng(19).integers(20,240,(256,256,3),dtype=np.uint8))
        for index in range(13):
            self.assertEqual(sorted(profile_id(index,e) for e in range(1,7)),list(range(6)))
        d4 = []
        for profile in range(6):
            clean, hard, trace = views.images(image,4,3,forced_profile=profile)
            c2,h2,_ = views.images(image,4,3,forced_profile=profile)
            self.assertEqual(clean.tobytes(),c2.tobytes())
            self.assertEqual(hard.tobytes(),h2.tobytes())
            self.assertNotEqual(clean.tobytes(),hard.tobytes())
            self.assertGreater(np.abs(np.asarray(clean,dtype=float)-np.asarray(hard,dtype=float)).mean(),5)
            d4.append(trace['d4'])
            if trace['profile']=='off_nadir':
                self.assertTrue((np.asarray(trace['corners'])>=0).all())
                self.assertTrue((np.asarray(trace['corners'])<=255).all())
                self.assertGreater(np.asarray(hard).min(),0)
        self.assertEqual(len(set(d4)),1)

    def test_persistent_workers_same_views_and_epoch_updates(self):
        cfg = json.loads(CONFIG.read_text())['data']
        cfg.update(batch_size=2,num_workers=2,persistent_workers=True,pin_memory=False)
        with tempfile.TemporaryDirectory() as directory:
            raw = fixture(Path(directory),count=4)
            ds = HardTargetDataset(raw/'train.csv',raw/'train',cfg,{3:0,9:1},42)
            worker_loader = make_loader(ds,cfg,42)
            serial_loader = make_loader(ds,{**cfg,'num_workers':0},42)
            try:
                for epoch in (1,2,6):
                    ds.set_epoch(epoch)
                    a,b = list(worker_loader),list(serial_loader)
                    for multi,serial in zip(a,b):
                        for left,right in zip(multi,serial):
                            torch.testing.assert_close(left,right,rtol=0,atol=0)
                        for profile,index in zip(multi[3],multi[4]):
                            self.assertEqual(profile.item(),profile_id(index.item(),epoch))
                        self.assertEqual(len(multi[0]),len(multi[2]))  # one common label per pair
            finally:
                if worker_loader._iterator is not None:
                    worker_loader._iterator._shutdown_workers()

    def test_js_math_symmetry_and_gradients(self):
        a,b = torch.randn(4,20,requires_grad=True),torch.randn(4,20,requires_grad=True)
        labels = torch.arange(4)
        total,terms = paired_loss(a,b,labels)
        p,q = a.softmax(1),b.softmax(1)
        m = (p+q)/2
        expected = .5*((p*(p.log()-m.log())).sum(1)+(q*(q.log()-m.log())).sum(1)).mean()
        torch.testing.assert_close(terms['js_loss'],expected)
        torch.testing.assert_close(total,.5*terms['clean_ce']+.5*terms['hard_ce']+3*expected)
        _,same = paired_loss(a,a,labels)
        torch.testing.assert_close(same['js_loss'],torch.zeros_like(same['js_loss']),atol=1e-7,rtol=0)
        terms['js_loss'].backward()
        self.assertGreater(a.grad.abs().sum(),0)
        self.assertGreater(b.grad.abs().sum(),0)
        extreme = torch.randn(4,20)*10000
        self.assertTrue(torch.isfinite(paired_loss(extreme,-extreme,labels)[0]))

    def test_configs_and_real_resnet_agmfa_preflight(self):
        for filename,epochs,blocks in [('resnet50_hard_target_shift',24,['layer3','layer4']),
                                       ('resnet50_agmfa_hard_target_shift',24,['layer3','layer4']),
                                       ('resnet50_agmfa_hard_target_shift_finetune',12,['layer2','layer3','layer4'])]:
            config = json.loads((ROOT/'configs'/f'{filename}.json').read_text())
            self.assertEqual(config['training']['epochs'],epochs)
            self.assertEqual(config['training']['scheduler']['T_max'],epochs)
            self.assertEqual(config['model']['unfreeze_blocks'],blocks)
            self.assertEqual(config['data']['augmentation'],{})
        for filename in ('resnet50_hard_target_shift','resnet50_agmfa_hard_target_shift'):
            report = check_hard_target_shift(ROOT/'configs'/f'{filename}.json')
            self.assertEqual(report['forward_images'],32)
            self.assertTrue(report['actual_model_forward_backward'])
            self.assertFalse(report['test_read'])

    def test_synthetic_pipeline_checkpoints_probabilities_counts_no_test(self):
        cfg = json.loads(CONFIG.read_text())
        cfg['model']['agmfa'].update(fusion_channels=32,norm_groups=4)
        cfg['training'].update(epochs=6,device='cpu')
        cfg['training']['scheduler']['T_max']=6
        cfg['data'].update(batch_size=3,num_workers=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            with patch('archive.src.models.agmfa.load_backbone', side_effect=lambda *a:(TinyResNet(),'IMAGENET1K_V2')), \
                 patch('archive.src.data.load_test', side_effect=AssertionError('No test access')):
                result = run_hard_target_shift(cfg,'hard_smoke',project_root=root,show_plots=False)
                restored,_ = load_checkpoint(result.paths['models']/'best.pt','cpu')
            self.assertEqual(result.history.training_mode.tolist(),['head_only']*3+['partial']*3)
            for name in ('best.pt','best_accuracy.pt','best_loss.pt','last.pt'):
                self.assertTrue((result.paths['models']/name).is_file())
            self.assertEqual((result.paths['models']/'best.pt').read_bytes(),(result.paths['models']/'best_accuracy.pt').read_bytes())
            counts = pd.read_csv(result.paths['logs']/'profile_counts.csv')
            self.assertTrue((counts.groupby('profile')['count'].sum()==6).all())
            self.assertTrue((counts.groupby('epoch')['count'].sum()==6).all())
            for name in ('val_probabilities.npz','val_best_loss_probabilities.npz','val_best_accuracy_probabilities.npz'):
                self.assertEqual(load_probabilities(result.paths['predictions']/name).probabilities.shape,(6,2))
            from archive.src.data import load_validation
            loader,classes = load_validation(root/'data/raw',cfg['data'])
            actual = probability_predictions(restored,loader,torch.device('cpu'),classes.label.tolist())
            expected = load_probabilities(result.paths['predictions']/'val_probabilities.npz')
            np.testing.assert_allclose(actual.probabilities,expected.probabilities,rtol=1e-6,atol=1e-7)
            self.assertTrue((result.paths['logs']/'learning_curves.png').is_file())
            with (root/'experiments.csv').open(newline='') as f:
                records = list(csv.DictReader(f))
            self.assertEqual(len(records),1)
            self.assertEqual(records[0]['submission_id'],'')

    def test_initialize_corrected_weights_only_new_optimizer(self):
        cfg = json.loads((ROOT/'configs/resnet50_agmfa_hard_target_shift_finetune.json').read_text())
        cfg['model']['agmfa'].update(fusion_channels=32,norm_groups=4)
        with tempfile.TemporaryDirectory() as directory, \
             patch('archive.src.models.agmfa.load_backbone', side_effect=lambda *a:(TinyResNet(),'IMAGENET1K_V2')):
            root = Path(directory)
            source_config = json.loads(CONFIG.read_text())
            source_config['model']['agmfa'].update(fusion_channels=32,norm_groups=4)
            source_model = build_model(source_config['model'],256,2,initialize_pretrained=False)
            path = root/cfg['training']['initialize_from']
            path.parent.mkdir(parents=True)
            torch.save({'config':source_config,'class_labels':[3,9],'epoch':17,
                        'model_state_dict':source_model.state_dict(),'optimizer_state_dict':{'obsolete':True},
                        'scheduler_state_dict':{'last_epoch':17}},path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            initialized,metadata = build_shift_model(cfg,[3,9],root,torch.device('cpu'))
            for name,value in source_model.state_dict().items():
                torch.testing.assert_close(value,initialized.state_dict()[name],rtol=0,atol=0)
            self.assertFalse(metadata['optimizer_resumed'])
            optimizer = make_optimizer(initialized,cfg['training']['optimizer'])
            self.assertFalse(optimizer.state)
            self.assertEqual([g['lr'] for g in optimizer.param_groups],[1e-5,5e-5])
            initialized.set_epoch(1)
            self.assertTrue(all(p.requires_grad==name.startswith(('fc.','layer2.','layer3.','layer4.'))
                                for name,p in initialized.backbone.named_parameters()))
            self.assertEqual(digest,hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':
    unittest.main()
