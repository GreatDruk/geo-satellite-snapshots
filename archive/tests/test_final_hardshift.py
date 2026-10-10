"""Final epoch, weighted origins, no target corruption, and two audited submissions."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from archive.src.experiment import load_checkpoint
from archive.src.final_hardshift import (TargetDomainDataset,OriginDataset,origin_collate,source_object_losses,
                                mixed_objective,run_final_hardshift)
from archive.src.hard_target_training import paired_loss
from archive.src.hard_target_shift import HardTargetDataset
from archive.src.models.pretrained import TransferClassifier
from archive.src.training.predict import load_probabilities
from archive.tests.test_hard_target_shift import fixture

ROOT=Path(__file__).resolve().parents[1]


class TinyBody(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.conv1=nn.Linear(3,4)
        self.bn1=nn.BatchNorm1d(4)
        for name in ('layer1','layer2','layer3','layer4'):
            setattr(self,name,nn.Sequential(nn.Linear(4,4),nn.ReLU()))
        self.fc=nn.Linear(4,classes)
    def forward(self,images):
        x=self.bn1(self.conv1(images.mean((2,3))))
        for name in ('layer1','layer2','layer3','layer4'):
            x=getattr(self,name)(x)
        return self.fc(x)


def tiny_factory(model_config,image_size,num_classes):
    return TransferClassifier(TinyBody(num_classes),model_config,'resnet50_agmfa')


def config(name):
    cfg=json.loads((ROOT/'configs'/f'resnet50_agmfa_hardshift_final_{name}.json').read_text())
    cfg['data'].update(num_workers=0,batch_size=3)
    cfg['training']['device']='cpu'
    cfg['final_hardshift']['expected_counts']={'train':4,'val':4}
    return cfg


class FinalHardShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_object_weights_and_single_origin_batches(self):
        clean,hard,target=[torch.randn(n,3,requires_grad=True) for n in (2,2,1)]
        sy,ty=torch.tensor([0,1]),torch.tensor([2])
        loss,terms,mass=mixed_objective(clean,hard,sy,target,ty)
        source,_=source_object_losses(clean,hard,sy)
        expected=(source.sum()+2*nn.functional.cross_entropy(target,ty,reduction='sum'))/4
        torch.testing.assert_close(loss,expected)
        self.assertEqual(mass,4)
        loss.backward()
        for value in (clean,hard,target):
            self.assertGreater(value.grad.abs().sum(),0)
        source_only,_,mass=mixed_objective(clean,hard,sy,target[:0],ty[:0])
        torch.testing.assert_close(source_only,paired_loss(clean,hard,sy)[0])
        self.assertEqual(mass,2)
        target_only,_,mass=mixed_objective(clean[:0],hard[:0],sy[:0],target,ty)
        torch.testing.assert_close(target_only,nn.functional.cross_entropy(target,ty))
        self.assertEqual(mass,2)

    def test_former_val_d4_only_origin_and_once_sampler(self):
        cfg=config('train_val')
        with tempfile.TemporaryDirectory() as directory:
            raw=fixture(Path(directory),count=4)
            target=TargetDomainDataset(raw,cfg['data'],{3:0,9:1},42)
            with patch('archive.src.hard_target_shift.apply_profile',side_effect=AssertionError('Target must never get hard profile')):
                target.set_epoch(3)
                a,label=target[0]
                b,label2=target[0]
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.assertEqual(label,label2)
            source=HardTargetDataset(raw/'train.csv',raw/'train',cfg['data'],{3:0,9:1},42)
            origin=OriginDataset(source,target)
            loader=DataLoader(origin,batch_size=3,shuffle=True,collate_fn=origin_collate)
            source_ids,target_ids=[],[]
            for batch in loader:
                source_ids.extend(batch['source_indices'].tolist())
                target_ids.extend(batch['target_indices'].tolist())
                self.assertEqual(len(batch['source_profiles']),len(batch['source_labels']))
                if batch['source_images'] is None:
                    self.assertIsNone(batch['source_hard'])
            self.assertEqual(sorted(source_ids),list(range(4)))
            self.assertEqual(sorted(target_ids),list(range(4)))

    def test_target_epoch_reaches_persistent_workers(self):
        cfg=config('train_val')
        with tempfile.TemporaryDirectory() as directory:
            raw=fixture(Path(directory),count=4)
            ds=TargetDomainDataset(raw,cfg['data'],{3:0,9:1},42)
            worker=DataLoader(ds,batch_size=2,num_workers=2,persistent_workers=True)
            serial=DataLoader(ds,batch_size=2,num_workers=0)
            try:
                for epoch in (1,2):
                    ds.set_epoch(epoch)
                    for parallel,single in zip(list(worker),list(serial)):
                        for left,right in zip(parallel,single):
                            torch.testing.assert_close(left,right,rtol=0,atol=0)
            finally:
                if worker._iterator is not None:
                    worker._iterator._shutdown_workers()

    def test_both_22_epoch_runs_and_test_gating(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture(root,count=4)
            with patch('archive.src.data.load_test',side_effect=AssertionError('No test during fitting')), \
                 patch('archive.src.final_hardshift.evaluate_model',side_effect=[(.1+i/100,.9 if i==0 else .25) for i in range(22)]):
                train=run_final_hardshift(config('train'),project_root=root,show_plots=False,model_factory=tiny_factory)
            self.assertEqual(train['history'].epoch.tolist(),list(range(1,23)))
            self.assertEqual(train['history'].training_mode.tolist(),['head_only']*3+['partial']*19)
            self.assertEqual(train['summary']['val_accuracy'],.25)  # never pick epoch1 peak.9
            self.assertFalse((train['paths']['models']/'best.pt').exists())
            state=torch.load(train['paths']['models']/'last.pt',weights_only=True)
            self.assertEqual(state['epoch'],22)
            self.assertEqual(state['val_accuracy'],.25)
            with patch('archive.src.data.load_test',side_effect=AssertionError('No test during fitting')), \
                 patch('archive.src.final_hardshift.evaluate_model',side_effect=AssertionError('No validation in refit')):
                refit=run_final_hardshift(config('train_val'),project_root=root,show_plots=False,model_factory=tiny_factory)
            h=refit['history']
            self.assertFalse(any(c.startswith('val_') for c in h.columns))
            self.assertTrue(h.source_samples.eq(4).all() and h.target_samples.eq(4).all())
            self.assertTrue(h.forward_images.eq(12).all())
            self.assertTrue((h.weighted_target_fraction==2/3).all())
            self.assertIsNone(refit['summary']['val_accuracy'])
            self.assertFalse((refit['paths']['predictions']/'val_probabilities.npz').exists())
            npz=load_probabilities(train['paths']['predictions']/'val_probabilities.npz')
            self.assertEqual(npz.probabilities.shape,(4,2))
            counts=pd.read_csv(refit['paths']['logs']/'profile_counts.csv')
            self.assertTrue((counts.groupby('epoch')['count'].sum()==4).all())
            self.assertTrue((counts[counts.epoch<=6].groupby('profile')['count'].sum()==4).all())
            with (root/'experiments.csv').open(newline='') as f:
                records=list(csv.DictReader(f))
            self.assertEqual(len(records),2)
            self.assertEqual(records[1]['val_accuracy'],'')
            with self.assertRaises(FileExistsError):
                run_final_hardshift(config('train'),project_root=root,model_factory=tiny_factory)



if __name__ == "__main__":
    unittest.main()
