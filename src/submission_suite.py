"""Audit final refits and replay four fixed test recipes with shared resumable caches."""
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import read_classes, load_test
from src.experiment import PROJECT_ROOT, load_checkpoint
from src.inference import file_hash, _data_config
from src.training.predict import (ProbabilityPredictions,
    ensemble_probabilities, save_probabilities, load_probabilities, save_submission)
from src.utils import get_device, set_seed, append_experiment

CHECKPOINTS = {
    'train101': 'artifacts/models/resnet101_partial_domain_shift_stronger_001/best.pt',
    'train50': 'artifacts/models/resnet50_partial_domain_shift_stronger_001/best.pt',
    'refit101': 'artifacts/models/resnet101_final_train_val_001/final.pt',
    'refit50': 'artifacts/models/resnet50_final_train_val_001/final.pt',
}
RECIPES = {'resnet101_train': ('train101',), 'resnet101_train_val': ('refit101',),
           'ensemble_train': ('train101', 'train50'),
           'ensemble_train_val': ('refit101', 'refit50')}
TTA = {'train101':'none', 'refit101':'none', 'train50':'d4', 'refit50':'d4'}


def _resolve(root, path):
    path=Path(path)
    return path if path.is_absolute() else root/path


def audit_refits(project_root=PROJECT_ROOT, *, raw_dir='data/raw', check_forward=True, checkpoints=None):
    """Check real source files/logs/journal/weights, without running train or val."""
    root=Path(project_root).resolve(); raw=_resolve(root,raw_dir)
    classes=read_classes(raw)
    journal=pd.read_csv(root/'experiments.csv',keep_default_na=False)
    rows=[]
    torch.set_num_threads(4)
    for architecture,epochs in [('resnet50',20),('resnet101',21)]:
        exp_id=f'{architecture}_final_train_val_001'
        logs=root/'artifacts/logs'/exp_id
        config=json.loads((logs/'config.json').read_text(encoding='utf-8'))
        expected=json.loads((root/f'configs/{architecture}_final_train_val.json').read_text(encoding='utf-8'))
        metadata=json.loads((logs/'metadata.json').read_text(encoding='utf-8'))
        summary=json.loads((logs/'summary.json').read_text(encoding='utf-8'))
        history=pd.read_csv(logs/'history.csv')
        key='refit50' if architecture=='resnet50' else 'refit101'
        source=_resolve(root,(checkpoints or CHECKPOINTS)[key])
        digest=file_hash(source)
        assert metadata['status']=='complete' and metadata['kind']=='final_train_val_refit'
        assert metadata['training_splits']=={'train':4999,'val':699}
        assert metadata['held_out_validation'] is False and metadata['test_used_for_fitting'] is False
        assert metadata['checkpoint_sha256']==digest
        for filename,sha in metadata['sources'].items(): assert file_hash(raw/filename)==sha, filename
        assert config['model']==expected['model'] and config['training']==expected['training']
        # Only the raw path may differ between Kaggle and this workspace.
        actual_data={k:v for k,v in config['data'].items() if k!='raw_dir'}
        expected_data={k:v for k,v in expected['data'].items() if k!='raw_dir'}
        assert actual_data==expected_data
        assert config['seed']==metadata['seed']==42
        assert history.epoch.tolist()==list(range(1,epochs+1))
        assert np.isfinite(history[['train_loss','train_accuracy','epoch_time_sec']].to_numpy()).all()
        assert history.train_loss.ge(0).all() and history.train_accuracy.between(0,1).all()
        assert history.train_samples.eq(5698).all() and not any(c.startswith('val_') for c in history)
        for index,rates in enumerate(history.learning_rates):
            rates=json.loads(rates)
            for group,initial in [('backbone',5e-5),('head',5e-4)]:
                lr=1e-6+(initial-1e-6)*(1+math.cos(math.pi*index/25))/2
                assert math.isclose(rates[group],lr,rel_tol=1e-10,abs_tol=1e-12)
        assert summary['epochs']==epochs and summary['val_accuracy'] is None
        record=journal.loc[journal.exp_id==exp_id]
        assert len(record)==1 and record.iloc[0].val_accuracy==''
        assert math.isclose(float(record.iloc[0].train_time_min),summary['train_time_min'],rel_tol=1e-10)
        weights=torch.load(source,map_location='cpu',weights_only=True)
        assert weights['final_refit'] is True and weights['epoch']==epochs and weights['val_accuracy'] is None
        assert weights['training_splits']==['train','val'] and weights['config']==config
        assert weights['class_labels']==classes.label.tolist() and weights['class_names']==classes.class_name.tolist()
        assert all(torch.isfinite(value).all().item() for value in weights['model_state_dict'].values())
        del weights
        if check_forward:
            model,_=load_checkpoint(source,'cpu',inference_only=True)
            with torch.inference_mode(): result=model(torch.zeros(1,3,256,256))
            assert result.shape==(1,20) and torch.isfinite(result).all()
            del model
        text=(logs/'training.log').read_text(encoding='utf-8')
        assert f'{epochs}/{epochs}' in text and 'no held-out val' in text
        rows.append({'exp_id':exp_id,'epochs':epochs,'train_samples':5698,
                     'last_train_accuracy':float(history.train_accuracy.iloc[-1]),
                     'last_train_loss':float(history.train_loss.iloc[-1]),
                     'train_time_min':summary['train_time_min'],'checkpoint_sha256':digest,'checks':'passed'})
    return pd.DataFrame(rows)


def _validate_test(prediction, ids, classes):
    prediction.validate()
    if (set(prediction.ids)!=set(ids) or set(prediction.class_labels)!=set(classes)
        or prediction.true_labels is not None):
        raise ValueError('Test cache has wrong IDs/classes or contains targets')


def run_submission_suite(run_id='four_test_submissions_001', *, project_root=PROJECT_ROOT,
                         raw_dir='data/raw', checkpoints=None, device='auto', batch_size=32,
                         num_workers=0, cpu_threads=4, resume=True):
    """Four CSV submissions; run each of four checkpoints once (2 none + 2 D4).

    Cache each batch with IDs/class order, resume only the same source/protocol.
    Never train, read validation labels, compute a test accuracy or upload to Kaggle.
    Journal rows describe inference; train_time_min/val_accuracy/public_score blank.
    """
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}',run_id):
        raise ValueError('Use a suite ID of at most 40 safe characters')
    if batch_size<1 or num_workers<0 or cpu_threads<1: raise ValueError('Invalid loader settings')
    root=Path(project_root).resolve(); raw=_resolve(root,raw_dir)
    sources=checkpoints if checkpoints is not None else CHECKPOINTS
    if set(sources)!=set(CHECKPOINTS): raise ValueError('Exactly the four named checkpoints required')
    sources={key:_resolve(root,path) for key,path in sources.items()}
    test=pd.read_csv(raw/'test.csv',dtype={'id':str})
    sample=pd.read_csv(raw/'sample_submission_example.csv',dtype={'id':str})
    classes=read_classes(raw).label.to_numpy()
    if (test.id.duplicated().any() or sample.id.duplicated().any() or list(sample.columns)!=['id','label']
        or set(test.id)!=set(sample.id)):
        raise ValueError('test/sample must contain the same unique IDs')
    selected=get_device(device)
    if selected.type=='cpu': torch.set_num_threads(cpu_threads)
    set_seed(42,deterministic=True,warn_only=True)
    source_specs={key:{'checkpoint':path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
                       'sha256':file_hash(path),'tta':TTA[key]} for key,path in sources.items()}
    protocol={'version':1,'sources':source_specs,'batch_size':batch_size,
              'test_csv_sha256':file_hash(raw/'test.csv'), 'classes_csv_sha256':file_hash(raw/'classes.csv'),
              'sample_sha256':file_hash(raw/'sample_submission_example.csv'),
              'recipes':{key:list(value) for key,value in RECIPES.items()},'weights':'equal probabilities'}
    logs=root/'artifacts/logs'/run_id; outputs=root/'artifacts/predictions'/run_id
    meta_path=logs/'metadata.json'
    if logs.exists() or outputs.exists():
        if not resume or not meta_path.exists(): raise FileExistsError('Existing suite; use resume or a new ID')
        previous=json.loads(meta_path.read_text(encoding='utf-8'))
        if previous['protocol']!=protocol: raise ValueError('Resume source/data/batch protocol changed')
    else:
        logs.mkdir(parents=True); outputs.mkdir(parents=True)
    metadata={'kind':'test_submission_suite','status':'running','protocol':protocol,'device':str(selected),
              'torch':str(torch.__version__),'training':False,'test_fitting':False,'seed':42}
    def update(): meta_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    def emit(message):
        print(message,flush=True)
        with (logs/'inference.log').open('a',encoding='utf-8') as f: f.write(message+'\n')
    update(); members={}; timings={}; started=time.perf_counter()
    try:
        for name,source in sources.items():
            cache=outputs/f'{name}.npz'; member_log=logs/f'{name}.json'
            if cache.exists() and member_log.exists():
                member_meta=json.loads(member_log.read_text(encoding='utf-8'))
                if member_meta['source']!=source_specs[name]: raise ValueError('Member cache source mismatch')
                if file_hash(cache)!=member_meta['probabilities_sha256']: raise ValueError('Cache hash mismatch')
                p=load_probabilities(cache); _validate_test(p,test.id,classes)
                members[name]=p; timings[name]=member_meta['inference_time_min']; emit(f'{name}: reused complete cache')
                continue
            model,checkpoint=load_checkpoint(source,selected,inference_only=True)
            if set(checkpoint['class_labels'])!=set(classes): raise ValueError('Checkpoint class mismatch')
            is_refit=bool(checkpoint.get('final_refit',False))
            if is_refit!=name.startswith('refit'): raise ValueError('Wrong train/refit checkpoint assignment')
            expected_arch='resnet101' if name.endswith('101') else 'resnet50'
            if checkpoint['config']['model']['name']!=expected_arch: raise ValueError('Wrong architecture assignment')
            data_config=_data_config(checkpoint,selected,batch_size,num_workers)
            loader=load_test(raw,data_config,42)
            chunks=outputs/'chunks'/name; chunks.mkdir(parents=True,exist_ok=True)
            parts=[]; member_started=time.perf_counter()
            for index,(images,_) in enumerate(loader):
                ids=loader.dataset.data.id.iloc[index*batch_size:index*batch_size+len(images)].to_numpy(dtype=str)
                chunk=chunks/f'{index:05d}.npz'
                if chunk.exists():
                    part=load_probabilities(chunk)
                    if not np.array_equal(part.ids,ids) or not np.array_equal(part.class_labels,checkpoint['class_labels']):
                        raise ValueError('Invalid resumed batch order/classes')
                    if part.true_labels is not None: raise ValueError('Test chunk contains targets')
                else:
                    # Existing inference primitive needs a sequential dataset/loader;
                    # perform these shared views directly for this resumable batch.
                    from src.training.predict import tta_views
                    with torch.inference_mode():
                        images=images.to(selected,non_blocking=selected.type=='cuda')
                        values=[model(view.contiguous()).float().softmax(1) for view in tta_views(images,TTA[name])]
                        probabilities=torch.stack(values).mean(0).cpu().numpy()
                    part=ProbabilityPredictions(ids,np.asarray(checkpoint['class_labels']),probabilities).validate()
                    # Publish only a complete archive; same-directory replace is atomic.
                    temporary=chunk.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream,ids=part.ids,class_labels=part.class_labels,probabilities=part.probabilities)
                    temporary.replace(chunk)
                parts.append(part.probabilities)
                if index==0 or (index+1)%10==0 or index+1==len(loader): emit(f'{name} {TTA[name]}: {index+1}/{len(loader)} batches')
            p=ProbabilityPredictions(test.id.to_numpy(dtype=str),np.asarray(checkpoint['class_labels']),np.concatenate(parts)).validate()
            _validate_test(p,test.id,classes)
            if cache.exists():
                old=load_probabilities(cache)
                if (not np.array_equal(old.ids,p.ids) or not np.array_equal(old.class_labels,p.class_labels)
                    or not np.allclose(old.probabilities,p.probabilities,atol=1e-7,rtol=0)):
                    raise ValueError('Orphan cache differs from verified chunks')
            else: save_probabilities(p,cache)
            timings[name]=(time.perf_counter()-member_started)/60
            member_log.write_text(json.dumps({'source':source_specs[name],'inference_time_min':timings[name],
                'probabilities_sha256':file_hash(cache),'final_refit':is_refit,
                'training_splits':['train','val'] if is_refit else ['train'],
                'epoch':checkpoint.get('epoch'),'config':checkpoint['config'],
                'class_labels':checkpoint['class_labels']},indent=2),encoding='utf-8')
            members[name]=p; del model,loader,checkpoint
        records=[]
        for recipe,keys in RECIPES.items():
            exp_id=f'{run_id}_{recipe}'
            p=ensemble_probabilities([members[key] for key in keys])
            cache=outputs/f'{recipe}.npz'
            if cache.exists():
                old=load_probabilities(cache)
                if (not np.array_equal(old.ids,p.ids) or not np.array_equal(old.class_labels,p.class_labels)
                    or not np.allclose(old.probabilities,p.probabilities,atol=1e-7,rtol=0)):
                    raise ValueError('Existing recipe probabilities differ')
            else: save_probabilities(p,cache)
            submission=root/'submissions'/f'{exp_id}.csv'
            expected=p.table().set_index('id').loc[sample.id].reset_index()
            if submission.exists():
                actual=pd.read_csv(submission,dtype={'id':str})
                if not actual.equals(expected): raise ValueError('Existing submission differs')
            else: save_submission(p.table(),raw/'sample_submission_example.csv',submission)
            journal=root/'experiments.csv'
            existing=pd.read_csv(journal,keep_default_na=False) if journal.exists() else pd.DataFrame()
            found=existing.loc[existing.exp_id==exp_id] if len(existing) else pd.DataFrame()
            if len(found)>1: raise ValueError('Duplicate submission journal rows')
            if len(found)==1:
                if found.iloc[0].submission_id!=exp_id: raise ValueError('Existing journal row is not this submission')
            else:
                append_experiment(journal,{'exp_id':exp_id,'model':'resnet101' if len(keys)==1 else 'ensemble_resnet101_resnet50',
                    'image_size':'256','augmentation':json.dumps({key:TTA[key] for key in keys}),'pretrained':True,
                    'val_accuracy':'','train_time_min':'','seed':42,'submission_id':exp_id,'public_score':'',
                    'notes':f'TEST INFERENCE ONLY; {recipe}; fixed equal probabilities; sources/sha in {run_id}/metadata.json; no test accuracy'})
            record={'submission_id':exp_id,'recipe':recipe,'images':len(p.ids),'submission':submission.relative_to(root).as_posix(),
                    'submission_sha256':file_hash(submission),'members':list(keys)}
            (logs/f'{recipe}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            records.append(record)
        result=pd.DataFrame(records); result.to_csv(logs/'submissions.csv',index=False)
        metadata.update(status='complete',inference_time_min=(time.perf_counter()-started)/60,
                        member_inference_time_min=timings,submissions=records)
        update(); return result
    except BaseException as error:
        metadata.update(status='failed',error=f'{type(error).__name__}: {error}'); update(); raise
