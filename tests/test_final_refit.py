"""Final refit must consume both labeled splits without validation selection."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.final_refit import refit_loader, run_final_refit
from src.experiment import load_checkpoint, run_experiment
from src.inference import file_hash, run_selected_inference


class FinalRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_ordinary_experiment_rejects_final_refit_before_loading_data(self):
        with self.assertRaisesRegex(ValueError,'run_final_refit'):
            run_experiment({'final_refit':{'training_splits':['train','val']}})

    def test_fixed_last_epoch_no_val_metric_or_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=root/'data/raw'; raw.mkdir(parents=True)
            pd.DataFrame({'label':[3,9],'class_name':['church','palace']}).to_csv(raw/'classes.csv',index=False)
            for split,ids,labels in [('train',['a.png','b.png'],[3,9]),('val',['c.png'],[9])]:
                (raw/split).mkdir()
                pd.DataFrame({'id':ids,'label':labels}).to_csv(raw/f'{split}.csv',index=False)
                for i,image_id in enumerate(ids):
                    Image.fromarray(np.full((4,4,3),80+80*i,dtype=np.uint8)).save(raw/split/image_id)
            config={'seed':42,'exp_id':'synthetic_final','final_refit':{'training_splits':['train','val']},
                    'data':{'raw_dir':'data/raw','image_size':4,'batch_size':2,'num_workers':0,'augmentation':{}},
                    'model':{'name':'mlp','layers':[{'type':'flatten'},{'type':'linear','out_features':'num_classes'}]},
                    'training':{'epochs':2,'device':'cpu','optimizer':{'name':'sgd','lr':.01},
                                'scheduler':{'name':'cosine','T_max':2}}}
            loader,classes,counts=refit_loader(raw,config['data'],42)
            self.assertEqual(counts,{'train':2,'val':1})
            self.assertEqual(len(loader.dataset),3)
            self.assertIs(loader.dataset.datasets[0].transform, loader.dataset.datasets[1].transform)
            raw_before={p:p.read_bytes() for p in raw.rglob('*') if p.is_file()}
            with patch('src.final_refit.load_test',side_effect=AssertionError('Test must not be read')):
                result=run_final_refit(config,project_root=root,show_plots=False)
            model,checkpoint=load_checkpoint(result['paths']['models']/'final.pt','cpu',inference_only=True)
            self.assertIsNone(checkpoint['val_accuracy'])
            self.assertEqual(checkpoint['epoch'],2)
            self.assertTrue(checkpoint['final_refit'])
            self.assertFalse((result['paths']['models']/'best.pt').exists())
            self.assertNotIn('val_accuracy',result['history'].columns)
            self.assertEqual(result['history'].train_samples.tolist(),[3,3])
            selection={'format_version':1,'selected_on':'val','recipe':'preselected_final',
                       'class_labels':[3,9],'val_csv_sha256':file_hash(raw/'val.csv'),
                       'members':[{'checkpoint':str(result['paths']['models']/'final.pt'),
                                   'sha256':file_hash(result['paths']['models']/'final.pt'),
                                   'weight':1.,'tta':'none'}]}
            with self.assertRaisesRegex(ValueError,'no held-out validation'):
                run_selected_inference(selection,'forbidden_val',project_root=root,split='val',device='cpu')
            with (root/'experiments.csv').open(newline='',encoding='utf-8') as stream:
                record=list(csv.DictReader(stream))[0]
            self.assertEqual(record['val_accuracy'],'')
            for path,before in raw_before.items(): self.assertEqual(path.read_bytes(),before)


if __name__=='__main__': unittest.main()
