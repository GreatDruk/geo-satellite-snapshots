"""Full/local view geometry, aligned fusion, replay and interrupted-run recovery."""
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

from src.local_inference import crop_boxes, LocalViews, fuse_global_local, run_local_diagnostic, analyze_saved_local
from src.inference import run_selected_inference
from src.models import build_model
from src.training.predict import ProbabilityPredictions, load_probabilities


class LocalInferenceTests(unittest.TestCase):
    def test_crops_use_original_pixels_and_include_full_frame(self):
        self.assertEqual(crop_boxes((256,256)),[(0,0,160,160),(96,0,256,160),(48,48,208,208),(0,96,160,256),(96,96,256,256)])
        array=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
        image=Image.fromarray(array)
        transform=LocalViews({'image_size':8},crop_size=5)
        views=transform(image)
        self.assertEqual(tuple(views.shape),(6,3,8,8))
        torch.testing.assert_close(views[0],transform.transform(image))
        torch.testing.assert_close(views[1],transform.transform(image.crop((0,0,5,5))))
        np.testing.assert_array_equal(np.array(image),array)
        with self.assertRaises(ValueError): crop_boxes((8,8),9)

    def test_max_fusion_aligns_and_normalizes_classwise_max(self):
        p=ProbabilityPredictions(np.array(['a','b']),np.array([4,11]),np.array([[.9,.1],[.2,.8]]),np.array([4,11]))
        reverse=ProbabilityPredictions(np.array(['b','a']),np.array([11,4]),np.array([[.3,.7],[.8,.2]]),np.array([11,4]))
        crops=[p,reverse,p,reverse,p]
        result=fuse_global_local(p,crops,'max')
        # a max = [.9,.8], b max=[.7,.8], not distributions before normalization.
        expected=.75*p.probabilities+.25*np.array([[.9/1.7,.8/1.7],[.7/1.5,.8/1.5]])
        np.testing.assert_allclose(result.probabilities,expected,atol=1e-7)
        self.assertTrue(np.allclose(result.probabilities.sum(1),1))
        reverse.true_labels=np.array([4,4])
        with self.assertRaises(ValueError): fuse_global_local(p,crops,'mean')

    def setup_source(self,root):
        raw=root/'data/raw'; (raw/'val').mkdir(parents=True)
        pd.DataFrame({'label':[4,11],'class_name':['dark','light']}).to_csv(raw/'classes.csv',index=False)
        pd.DataFrame({'id':['a.png','b.png','c.png'],'label':[4,11,4]}).to_csv(raw/'val.csv',index=False)
        for name,color in [('a.png',0),('b.png',255),('c.png',16)]:
            Image.fromarray(np.full((8,8,3),color,dtype=np.uint8)).save(raw/'val'/name)
        config={'seed':42,'data':{'image_size':8},'model':{'name':'mlp','layers':[{'type':'flatten'},{'type':'linear','out_features':'num_classes'}]}}
        model=build_model(config['model'],8,2)
        with torch.no_grad():
            model[1].weight[0].fill_(-.1);model[1].weight[1].fill_(.1);model[1].bias.zero_()
        checkpoint=root/'best.pt'
        saved={'config':config,'class_labels':[4,11],'epoch':1,'model_state_dict':model.state_dict()}
        torch.save(saved,checkpoint)
        return raw,checkpoint,model,saved

    def test_suite_replay_and_completed_cache_do_not_fit_or_touch_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,checkpoint,_,_=self.setup_source(root)
            with contextlib.redirect_stdout(io.StringIO()):
                suite=run_local_diagnostic(checkpoint,'local',project_root=root,device='cpu',crop_size=5,batch_size=1,cpu_threads=1)
            self.assertEqual(len(suite['diagnostic']),3)
            self.assertEqual(len(suite['comparison']),10)
            reviewed=analyze_saved_local('local',project_root=root)
            self.assertEqual(reviewed['summary']['best_correct'],3)
            self.assertTrue((reviewed['paths']['logs']/'selected_class_metrics.csv').exists())
            self.assertFalse((root/'experiments.csv').exists())
            self.assertFalse((raw/'train.csv').exists())
            self.assertFalse((raw/'test.csv').exists())
            selection={**suite['selection'],'format_version':2,'method':'global_local','local_pool':'max','global_weight':.75,'crop_size':5}
            replay=run_selected_inference(selection,'replay',project_root=root,device='cpu',batch_size=2)
            expected=load_probabilities(suite['paths']['predictions']/'global_local_max.npz')
            np.testing.assert_allclose(replay['predictions'].probabilities,expected.probabilities,atol=1e-7)
            # Test replay is explicit and respects sample order and placeholder targets.
            (raw/'test').mkdir()
            for name in ['a.png','b.png']:
                (raw/'test'/name).write_bytes((raw/'val'/name).read_bytes())
            pd.DataFrame({'id':['a.png','b.png'],'label':[-1,-1]}).to_csv(raw/'test.csv',index=False)
            pd.DataFrame({'id':['b.png','a.png'],'label':[0,0]}).to_csv(raw/'sample_submission_example.csv',index=False)
            test_replay=run_selected_inference(selection,'test_replay',project_root=root,device='cpu',batch_size=2,split='test')
            self.assertIsNone(test_replay['predictions'].true_labels)
            self.assertEqual(pd.read_csv(test_replay['paths']['submission']).id.tolist(),['b.png','a.png'])
            with patch('src.local_inference.load_checkpoint',side_effect=AssertionError('complete cache avoids forwards')):
                cached=run_local_diagnostic(checkpoint,'local',project_root=root,device='cpu',crop_size=5,batch_size=1,resume=True)
            self.assertEqual(cached['selection'],suite['selection'])
            with self.assertRaises(ValueError):
                run_local_diagnostic(checkpoint,'local',project_root=root,device='cpu',crop_size=4,batch_size=1,resume=True)

    def test_interrupted_run_reuses_completed_batch_and_preserves_input_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,checkpoint,model,saved=self.setup_source(root)
            original=model.forward
            calls=[0]
            def forward(x):
                calls[0]+=1
                if calls[0]==14: raise RuntimeError('simulated interruption')
                return original(x)
            model.forward=forward
            with patch('src.local_inference.load_checkpoint',return_value=(model,saved)),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'simulated'):
                    run_local_diagnostic(checkpoint,'partial',project_root=root,device='cpu',crop_size=5,batch_size=1,cpu_threads=1)
            self.assertEqual(len(list((root/'artifacts/predictions/partial/chunks').glob('*.npz'))),1)
            calls[0]=0;model.forward=original
            with patch.object(model,'forward',wraps=original) as count,patch('src.local_inference.load_checkpoint',return_value=(model,saved)),contextlib.redirect_stdout(io.StringIO()):
                resumed=run_local_diagnostic(checkpoint,'partial',project_root=root,device='cpu',crop_size=5,batch_size=1,cpu_threads=1,resume=True)
            self.assertEqual(count.call_count,26)  # 2 remaining images x13, no repeat of batch1
            self.assertEqual(resumed['diagnostic'].id.tolist(),['a.png','b.png','c.png'])


if __name__=='__main__': unittest.main()
