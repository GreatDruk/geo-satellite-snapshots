"""Average->avgmax initialization preserves logits and survives train/checkpoint replay."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

from archive.src.models import build_model
from archive.src.models.initialization import initialize_from_checkpoint
from archive.src.experiment import run_experiment, load_checkpoint


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,4,3,padding=1)
        for i in range(1,5):setattr(self,f'layer{i}',nn.Sequential(nn.Conv2d(4,4,1),nn.ReLU()))
        self.avgpool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(4,1000)
    def forward(self,x):
        x=self.conv1(x)
        for i in range(1,5):x=getattr(self,f'layer{i}')(x)
        return self.fc(self.avgpool(x).flatten(1))


class AvgMaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def config(self,pooling='average'):
        return {'seed':42,'data':{'image_size':8,'batch_size':2,'num_workers':0},
                'model':{'name':'resnet18','pretrained':True,'mode':'partial','unfreeze_blocks':['layer3','layer4'],
                         'pooling':pooling,'head_dropout':.3,'warmup_epochs':1,'freeze_bn':True},
                'training':{'epochs':2,'log_every':1,'device':'cpu','optimizer':{'name':'adamw','lr':.001}}}

    def build(self,config):
        with patch('archive.src.models.pretrained.load_backbone',return_value=(TinyBackbone(),'DEFAULT')):
            return build_model(config['model'],8,2,initialize_pretrained=False)

    def test_preserves_logits_and_new_max_weights_receive_gradient(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=self.config();old=self.build(cfg).eval();new=self.build(self.config('avgmax')).eval()
            path=Path(folder)/'old.pt'
            torch.save({'config':cfg,'class_labels':[4,11],'epoch':21,'model_state_dict':old.state_dict()},path)
            meta=initialize_from_checkpoint(new,path,[4,11],cfg['data'])
            self.assertTrue(meta['head_expansion']);self.assertFalse(meta['optimizer_resumed'])
            x=torch.randn(3,3,8,8)
            with torch.no_grad():torch.testing.assert_close(new(x),old(x),atol=1e-6,rtol=1e-6)
            self.assertEqual(new.backbone.fc[1].weight.shape,(2,8))
            new.set_epoch(2);new(x).sum().backward()
            self.assertGreater(new.backbone.fc[1].weight.grad[:,4:].abs().sum().item(),0)
            with self.assertRaisesRegex(ValueError,'ordered classes'):
                initialize_from_checkpoint(new,path,[11,4],cfg['data'])

    def test_fine_tuning_records_source_and_reloads_without_source_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'data/raw'
            for split in ['train','val']:
                (raw/split).mkdir(parents=True)
                pd.DataFrame({'id':['a.png','b.png'],'label':[4,11]}).to_csv(raw/f'{split}.csv',index=False)
                for name,color in [('a.png',0),('b.png',255)]:Image.fromarray(np.full((8,8,3),color,dtype=np.uint8)).save(raw/split/name)
            pd.DataFrame({'label':[4,11],'class_name':['a','b']}).to_csv(raw/'classes.csv',index=False)
            cfg=self.config();old=self.build(cfg)
            source=root/'source.pt'
            torch.save({'config':cfg,'class_labels':[4,11],'epoch':21,'model_state_dict':old.state_dict()},source)
            new_cfg=self.config('avgmax');new_cfg['training']['initialize_from']='source.pt'
            with patch('archive.src.models.pretrained.load_backbone',return_value=(TinyBackbone(),'DEFAULT')) as loader,contextlib.redirect_stdout(io.StringIO()):
                run=run_experiment(new_cfg,'avgmax',project_root=root,show_plots=False)
            self.assertFalse(loader.call_args.args[2]) # no ImageNet download when using trained weights
            meta=json.loads((run.paths['logs']/'metadata.json').read_text(encoding='utf-8'))
            self.assertTrue(meta['initialization']['head_expansion'])
            self.assertEqual(meta['initialization']['source_epoch'],21)
            self.assertEqual(pd.read_csv(root/'experiments.csv').shape[0],1)
            self.assertTrue((run.paths['predictions']/'val_probabilities.npz').exists())
            source.unlink()
            with patch('archive.src.models.pretrained.load_backbone',return_value=(TinyBackbone(),'DEFAULT')):
                restored,_=load_checkpoint(run.paths['models']/'best.pt','cpu',inference_only=True)
            self.assertEqual(tuple(restored(torch.zeros(2,3,8,8)).shape),(2,2))


if __name__=='__main__': unittest.main()
