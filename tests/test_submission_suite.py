"""Four fixed test recipes, shared TTA caches, crash recovery and journal rows."""
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

from src.submission_suite import run_submission_suite, CHECKPOINTS, TTA
from src.training.predict import load_probabilities, ensemble_probabilities, tta_views


@contextlib.contextmanager
def workspace_temp():
    # Windows sandbox Temp denies atomic rename; use a verified workspace path.
    parent=(Path(__file__).resolve().parents[1]/'artifacts').resolve()
    with tempfile.TemporaryDirectory(dir=parent,prefix='tmp_submission_test_') as directory:
        assert Path(directory).resolve().is_relative_to(parent)
        yield directory


class Tiny(torch.nn.Module):
    def __init__(self, name, fail=False):
        super().__init__(); self.name=name; self.calls=0; self.fail=fail
    def forward(self,x):
        self.calls+=1
        if self.fail and self.calls==2: raise RuntimeError('interrupted')
        # Orientation-sensitive signal for testing probability averaging.
        signal=x[:,0,0,0]+(.2 if 'refit' in self.name else 0)
        return torch.stack([signal,-signal],dim=1)


class SubmissionSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setup_data(self,root):
        raw=root/'data/raw'; raw.mkdir(parents=True); (raw/'test').mkdir()
        ids=['b.png','a.png','c.png']
        pd.DataFrame({'id':ids}).to_csv(raw/'test.csv',index=False)
        pd.DataFrame({'id':ids[::-1],'label':[0,0,0]}).to_csv(raw/'sample_submission_example.csv',index=False)
        pd.DataFrame({'label':[3,9],'class_name':['church','palace']}).to_csv(raw/'classes.csv',index=False)
        for i,image_id in enumerate(ids):
            array=np.arange(48,dtype=np.uint8).reshape(4,4,3)*(i+1)
            Image.fromarray(array).save(raw/'test'/image_id)
        sources={}
        for name in CHECKPOINTS:
            path=root/f'{name}.pt'; path.write_bytes(name.encode()); sources[name]=path
        return raw,sources

    def loader(self,models):
        def load(path,device,inference_only):
            name=Path(path).stem; model=models[name]
            return model.eval(),{'final_refit':name.startswith('refit'),'class_labels':[3,9],
                'config':{'seed':42,'model':{'name':'resnet101' if name.endswith('101') else 'resnet50'},
                          'data':{'image_size':4,'normalization':{'mean':[0,0,0],'std':[1,1,1]}}}}
        return load

    def test_four_csvs_shared_forwards_alignment_journal_and_completed_resume(self):
        with workspace_temp() as directory:
            root=Path(directory); raw,sources=self.setup_data(root)
            models={name:Tiny(name) for name in CHECKPOINTS}
            before={p:p.read_bytes() for p in raw.rglob('*') if p.is_file()}
            with patch('src.submission_suite.load_checkpoint',side_effect=self.loader(models)), contextlib.redirect_stdout(io.StringIO()):
                table=run_submission_suite('suite',project_root=root,checkpoints=sources,device='cpu',batch_size=2)
            self.assertEqual(len(table),4)
            self.assertEqual({k:m.calls for k,m in models.items()}, {'train101':2,'train50':16,'refit101':2,'refit50':16})
            for row in table.itertuples():
                csv=pd.read_csv(root/row.submission)
                self.assertEqual(list(csv.columns),['id','label'])
                self.assertEqual(csv.id.tolist(),['c.png','a.png','b.png'])
                self.assertTrue(csv.label.isin([3,9]).all())
            journal=pd.read_csv(root/'experiments.csv',keep_default_na=False)
            self.assertEqual(len(journal),4)
            self.assertTrue(journal.val_accuracy.eq('').all())
            self.assertTrue(journal.train_time_min.eq('').all())
            output=root/'artifacts/predictions/suite'
            expected=ensemble_probabilities([load_probabilities(output/'train101.npz'),load_probabilities(output/'train50.npz')])
            np.testing.assert_allclose(load_probabilities(output/'ensemble_train.npz').probabilities,expected.probabilities)
            # Independent manual D4 check on the original image batch.
            x=torch.from_numpy(np.array(Image.open(raw/'test/b.png'),dtype=np.float32)/255).permute(2,0,1).unsqueeze(0)
            manual=torch.stack([Tiny('train50')(v).softmax(1) for v in tta_views(x,'d4')]).mean(0).numpy()[0]
            np.testing.assert_allclose(load_probabilities(output/'train50.npz').probabilities[0],manual,atol=1e-7)
            with patch('src.submission_suite.load_checkpoint',side_effect=AssertionError('No forwards on completed resume')), contextlib.redirect_stdout(io.StringIO()):
                run_submission_suite('suite',project_root=root,checkpoints=sources,device='cpu',batch_size=2)
            self.assertEqual(len(pd.read_csv(root/'experiments.csv')),4)
            for p,value in before.items(): self.assertEqual(p.read_bytes(),value)
            with self.assertRaisesRegex(ValueError,'protocol changed'):
                run_submission_suite('suite',project_root=root,checkpoints=sources,device='cpu',batch_size=1)

    def test_resume_after_interruption_reuses_finished_chunks(self):
        with workspace_temp() as directory:
            root=Path(directory); _,sources=self.setup_data(root)
            models={name:Tiny(name,fail=name=='train101') for name in CHECKPOINTS}
            with patch('src.submission_suite.load_checkpoint',side_effect=self.loader(models)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    run_submission_suite('resume',project_root=root,checkpoints=sources,device='cpu',batch_size=2)
            first=root/'artifacts/predictions/resume/chunks/train101/00000.npz'
            before=first.read_bytes()
            models={name:Tiny(name) for name in CHECKPOINTS}
            with patch('src.submission_suite.load_checkpoint',side_effect=self.loader(models)), contextlib.redirect_stdout(io.StringIO()):
                result=run_submission_suite('resume',project_root=root,checkpoints=sources,device='cpu',batch_size=2)
            self.assertEqual(models['train101'].calls,1)
            self.assertEqual(first.read_bytes(),before)
            self.assertEqual(len(result),4)

    def test_wrong_refit_checkpoint_is_rejected(self):
        with workspace_temp() as directory:
            root=Path(directory); _,sources=self.setup_data(root)
            models={name:Tiny(name) for name in CHECKPOINTS}
            original=self.loader(models)
            def wrong(*args,**kwargs):
                model,checkpoint=original(*args,**kwargs); checkpoint['final_refit']=True
                return model,checkpoint
            with patch('src.submission_suite.load_checkpoint',side_effect=wrong):
                with self.assertRaisesRegex(ValueError,'assignment'):
                    run_submission_suite('wrong',project_root=root,checkpoints=sources,device='cpu',batch_size=2)


if __name__=='__main__': unittest.main()
