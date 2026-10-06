"""Read-only audit of a completed four-submission suite; no image inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SUITE='submissions_001'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def read_prediction(path):
    with np.load(path,allow_pickle=False) as archive:
        assert 'true_labels' not in archive.files, f'Test targets present: {path}'
        ids=archive['ids']; labels=archive['class_labels']; values=archive['probabilities']
    assert ids.ndim==labels.ndim==1 and len(set(ids))==len(ids) and len(set(labels))==len(labels)
    assert values.shape==(len(ids),len(labels)) and np.isfinite(values).all()
    assert ((values>=0)&(values<=1)).all() and np.allclose(values.sum(1),1,atol=1e-5)
    return ids,labels,values


def audit():
    logs=ROOT/'artifacts/logs'/SUITE; outputs=ROOT/'artifacts/predictions'/SUITE
    metadata=json.loads((logs/'metadata.json').read_text(encoding='utf-8'))
    protocol=metadata['protocol']
    assert metadata['status']=='complete' and metadata['kind']=='test_submission_suite'
    assert metadata['training'] is False and metadata['test_fitting'] is False and metadata['seed']==42
    raw=ROOT/'data/raw'
    for filename,key in [('test.csv','test_csv_sha256'),('classes.csv','classes_csv_sha256'),
                         ('sample_submission_example.csv','sample_sha256')]:
        assert sha(raw/filename)==protocol[key], filename
    test=pd.read_csv(raw/'test.csv',dtype={'id':str})
    sample=pd.read_csv(raw/'sample_submission_example.csv',dtype={'id':str})
    classes=pd.read_csv(raw/'classes.csv')
    assert len(test)==len(sample)==8299 and len(classes)==20
    assert test.id.is_unique and sample.id.is_unique and set(test.id)==set(sample.id)
    members={}; member_checks=[]
    for name,source in protocol['sources'].items():
        assert sha(ROOT/source['checkpoint'])==source['sha256'], name
        details=json.loads((logs/f'{name}.json').read_text(encoding='utf-8'))
        assert details['source']==source and details['final_refit']==name.startswith('refit')
        assert details['training_splits']==(['train','val'] if name.startswith('refit') else ['train'])
        assert details['config']['model']['name']==('resnet101' if name.endswith('101') else 'resnet50')
        assert source['tta']==('none' if name.endswith('101') else 'd4')
        assert details['config']['data']['image_size']==256 and details['config']['seed']==42
        expected_epoch={'train101':21,'train50':20,'refit101':21,'refit50':20}[name]
        assert details['epoch']==expected_epoch
        assert sha(outputs/f'{name}.npz')==details['probabilities_sha256']
        ids,labels,values=read_prediction(outputs/f'{name}.npz')
        assert np.array_equal(ids,test.id.to_numpy()) and set(labels)==set(classes.label)
        assert np.array_equal(labels,np.asarray(details['class_labels']))
        chunks=sorted((outputs/'chunks'/name).glob('*.npz'))
        checked=0
        if chunks:
            assert len(chunks)==int(np.ceil(len(ids)/protocol['batch_size']))
            for index,path in enumerate(chunks):
                chunk_ids,chunk_labels,chunk_values=read_prediction(path)
                start=index*protocol['batch_size']; end=min(start+protocol['batch_size'],len(ids))
                assert path.name==f'{index:05d}.npz'
                assert np.array_equal(chunk_ids,ids[start:end]) and np.array_equal(chunk_labels,labels)
                assert np.array_equal(chunk_values,values[start:end])
                checked+=1
        members[name]=(ids,labels,values)
        member_checks.append({'member':name,'images':len(ids),'tta':source['tta'],'epoch':details['epoch'],
                              'chunks_checked':checked,'inference_time_min':details['inference_time_min']})
    expected_recipes={'resnet101_train':['train101'],'resnet101_train_val':['refit101'],
                     'ensemble_train':['train101','train50'],'ensemble_train_val':['refit101','refit50']}
    assert protocol['recipes']==expected_recipes and protocol['weights']=='equal probabilities'
    journal=pd.read_csv(ROOT/'experiments.csv',keep_default_na=False)
    listing=pd.read_csv(logs/'submissions.csv')
    assert len(listing)==len(metadata['submissions'])==4
    checks=[]
    for record in metadata['submissions']:
        recipe=record['recipe']; keys=expected_recipes[recipe]
        assert record['members']==keys and record['images']==8299
        ids,labels,values=read_prediction(outputs/f'{recipe}.npz')
        assert np.array_equal(ids,test.id.to_numpy()) and set(labels)==set(classes.label)
        expected=np.zeros_like(values,dtype=np.float64)
        for key in keys:
            member_ids,member_labels,member_values=members[key]
            rows=pd.Index(member_ids).get_indexer(ids); columns=pd.Index(member_labels).get_indexer(labels)
            expected+=member_values[rows][:,columns].astype(np.float64)/len(keys)
        expected=expected.astype(np.float32)
        assert np.array_equal(values,expected), f'Wrong single/ensemble arithmetic: {recipe}'
        prediction=pd.DataFrame({'id':ids,'label':labels[values.argmax(1)]}).set_index('id').loc[sample.id].reset_index()
        csv_path=ROOT/record['submission']; actual=pd.read_csv(csv_path,dtype={'id':str})
        assert actual.equals(prediction), f'CSV differs from probabilities/sample order: {recipe}'
        assert list(actual.columns)==['id','label'] and len(actual)==8299 and actual.id.is_unique
        assert actual.label.isin(classes.label).all() and not actual.isna().any().any()
        assert sha(csv_path)==record['submission_sha256']
        local_record=json.loads((logs/f'{recipe}.json').read_text(encoding='utf-8'))
        assert local_record==record
        row=journal.loc[journal.exp_id==record['submission_id']]
        assert len(row)==1 and row.iloc[0].submission_id==record['submission_id']
        assert row.iloc[0].val_accuracy=='' and row.iloc[0].train_time_min=='' and int(row.iloc[0].seed)==42
        assert row.iloc[0].notes.startswith('TEST INFERENCE ONLY')
        assert json.loads(row.iloc[0].augmentation)=={key:protocol['sources'][key]['tta'] for key in keys}
        assert listing.loc[listing.submission_id==record['submission_id']].iloc[0].submission_sha256==record['submission_sha256']
        checks.append({'recipe':recipe,'images':8299,'checks':'passed','submission':record['submission'],
                       'public_score':row.iloc[0].public_score,'mean_confidence':float(values.max(1).mean())})
    # Log progress must reach the final batch for all four members.
    text=(logs/'inference.log').read_text(encoding='utf-8')
    batches=int(np.ceil(len(test)/protocol['batch_size']))
    for name in members:
        assert f'{name} {protocol["sources"][name]["tta"]}: {batches}/{batches} batches' in text
    changes=[]
    for left,right in [('resnet101_train','resnet101_train_val'),('ensemble_train','ensemble_train_val'),
                       ('resnet101_train','ensemble_train'),('resnet101_train_val','ensemble_train_val')]:
        a_ids,a_labels,a=read_prediction(outputs/f'{left}.npz')
        b_ids,b_labels,b=read_prediction(outputs/f'{right}.npz')
        b=b[pd.Index(b_ids).get_indexer(a_ids)]
        different=int((a_labels[a.argmax(1)]!=b_labels[b.argmax(1)]).sum())
        changes.append({'left':left,'right':right,'different_labels':different})
    result={'suite':SUITE,'status':'passed','device':metadata['device'],'batch_size':protocol['batch_size'],
            'inference_time_min':metadata['inference_time_min'],'members':member_checks,
            'submissions':checks,'differences':changes,'test_accuracy_available':False}
    return result


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/test_submissions_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
