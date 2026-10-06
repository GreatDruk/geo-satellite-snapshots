"""Label-free diagnostics, alignment, ranking and deterministic random sampling."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.test_analysis import RECIPES,test_case_table,select_cases,analysis_tables,load_test_analysis
from src.training.predict import ProbabilityPredictions,save_probabilities
from src.inference import file_hash


class TestAnalysisTests(unittest.TestCase):
    def test_loader_aligns_raw_ids_and_classes_and_returns_unlabeled_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=root/'data/raw'; raw.mkdir(parents=True)
            ids=np.array(['a','b','c']); labels=np.array([3,9,5])
            pd.DataFrame({'id':ids}).to_csv(raw/'test.csv',index=False)
            pd.DataFrame({'label':labels,'class_name':['church','palace','commercial']}).to_csv(raw/'classes.csv',index=False)
            logs=root/'artifacts/logs/suite'; outputs=root/'artifacts/predictions/suite'
            logs.mkdir(parents=True); outputs.mkdir(parents=True)
            canonical=np.array([[.7,.2,.1],[.1,.8,.1],[.1,.2,.7]],dtype=np.float32)
            skew=ProbabilityPredictions(ids[::-1],labels[::-1],canonical[::-1,::-1])
            recipes={'resnet101_train':['train101'],'resnet101_train_val':['refit101'],
                     'ensemble_train':['train101','train50'],'ensemble_train_val':['refit101','refit50']}
            for name in ['train101','train50','refit101','refit50']:
                path=outputs/f'{name}.npz';save_probabilities(skew,path)
                (logs/f'{name}.json').write_text(json.dumps({'probabilities_sha256':file_hash(path)}),encoding='utf-8')
            for name in RECIPES:save_probabilities(skew,outputs/f'{name}.npz')
            meta={'status':'complete','protocol':{'test_csv_sha256':file_hash(raw/'test.csv'),
                  'classes_csv_sha256':file_hash(raw/'classes.csv'),
                  'sources':{name:{} for name in ['train101','train50','refit101','refit50']},'recipes':recipes}}
            (logs/'metadata.json').write_text(json.dumps(meta),encoding='utf-8')
            members,classes,sources=load_test_analysis(root,'suite',include_sources=True)
            for p in [*members.values(),*sources.values()]:
                np.testing.assert_array_equal(p.ids,ids)
                np.testing.assert_array_equal(p.class_labels,labels)
                np.testing.assert_array_equal(p.probabilities,canonical)
                self.assertIsNone(p.true_labels)
                self.assertFalse(p.ids.dtype.hasobject)

    def inputs(self):
        ids=np.array(['a','b','c']); labels=np.array([3,9,5]); members={}
        for i,name in enumerate(RECIPES):
            values=np.array([[.51,.49,0],[.99,.005,.005],[.1,.1,.8]])
            if i%2: values[1]=[.005,.99,.005]
            members[name]=ProbabilityPredictions(ids,labels,values)
        classes=pd.DataFrame({'label':labels,'class_name':['church','palace','commercial']})
        return members,classes

    def test_uncertainty_disagreement_js_and_confident_routing(self):
        members,classes=self.inputs(); table=test_case_table(members,classes)
        self.assertEqual(select_cases(table,'uncertain',limit=1).id.tolist(),['a'])
        self.assertEqual(select_cases(table,'disagreement').id.tolist(),['b'])
        self.assertEqual(select_cases(table,'confident_disagreement').id.tolist(),['b'])
        self.assertEqual(select_cases(table,'confident_disagreement',compare=RECIPES[:2]).id.tolist(),['b'])
        self.assertEqual(table.distinct_answers.tolist(),[1,2,1])
        self.assertGreater(table.js_divergence.iloc[1],.6)
        self.assertLess(table.js_divergence.iloc[0],1e-10)
        self.assertEqual(select_cases(table,'pair',pair=['church','palace']).id.tolist(),['a','b'])

    def test_random_sampling_repeatable_and_changes_are_not_errors(self):
        members,classes=self.inputs(); table=test_case_table(members,classes)
        pd.testing.assert_frame_equal(select_cases(table,'random',limit=2),select_cases(table,'random',limit=2))
        tables=analysis_tables(table,classes)
        self.assertEqual(len(tables['pairwise']),6)
        for name in RECIPES:
            self.assertEqual(tables['class_counts'].loc[tables['class_counts'].recipe==name,'count'].sum(),3)
        self.assertFalse(any('correct' in column or 'true' in column or 'error' in column for column in table))

    def test_rejects_truth_or_unaligned_arrays(self):
        members,classes=self.inputs()
        members[RECIPES[0]].true_labels=np.array([3,3,5])
        with self.assertRaises(ValueError): test_case_table(members,classes)
        members,classes=self.inputs(); members[RECIPES[1]].ids=members[RECIPES[1]].ids[::-1]
        with self.assertRaises(ValueError): test_case_table(members,classes)


if __name__=='__main__': unittest.main()
