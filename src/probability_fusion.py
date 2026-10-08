"""One fixed 50/50 fusion of saved probabilities; no training or weight search."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from src.training.predict import (load_probabilities,save_probabilities,ensemble_probabilities,save_submission)

MODELS=('agmfa','old_ensemble','blend_50_50')
GROUPS={'church/palace':['church','palace'], 'railway/station':['railway','railway_station'],
        'lake/wetland':['lake','wetland'], 'courts':['tennis_court','basketball_court']}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(prediction):
    values=prediction.probabilities
    true_index=pd.Index(prediction.class_labels).get_indexer(prediction.true_labels)
    if (true_index<0).any():
        raise ValueError('Unknown validation targets')
    correct=prediction.class_labels[values.argmax(1)]==prediction.true_labels
    confidence=values.max(1)
    errors=~correct
    one_hot=np.eye(len(prediction.class_labels))[true_index]
    return {'correct':int(correct.sum()),'errors':int(errors.sum()),'accuracy':float(correct.mean()),
            'nll':float(-np.log(np.clip(values[np.arange(len(values)),true_index],1e-30,1)).mean()),
            'brier':float(((values-one_hot)**2).sum(1).mean()),
            'mean_wrong_confidence':float(confidence[errors].mean()) if errors.any() else None,
            'confident_errors_90':int((errors&(confidence>=.9)).sum()),
            'confident_errors_95':int((errors&(confidence>=.95)).sum())}


def analyze_fusion(agmfa_path,old_path,classes_csv,*,val_dir=None):
    agmfa,old=load_probabilities(agmfa_path),load_probabilities(old_path)
    if agmfa.true_labels is None or old.true_labels is None:
        raise ValueError('Val analysis requires true labels; never evaluate refit on training val')
    # Align ID AND class-column order, requiring identical ground truth.
    old=ensemble_probabilities([agmfa,old],[0,1])
    blend=ensemble_probabilities([agmfa,old],[.5,.5])
    predictions=dict(zip(MODELS,(agmfa,old,blend)))
    classes=pd.read_csv(classes_csv)
    if set(classes.label)!=set(agmfa.class_labels):
        raise ValueError('Class mapping does not match probability files')
    frame=pd.DataFrame({'id':agmfa.ids,'true_label':agmfa.true_labels})
    for name,prediction in predictions.items():
        table=prediction.table(include_scores=True)
        for column in ('label','confidence','margin','entropy'):
            frame[f'{name}_{column}']=table[column].to_numpy()
        frame[f'{name}_correct']=table.label.to_numpy()==frame.true_label.to_numpy()
    a,o,b=[frame[f'{name}_correct'] for name in MODELS]
    categories=[('both_correct',a&o),('both_wrong',~a&~o),('only_agmfa_correct',a&~o),
                ('only_old_correct',~a&o),('blend_fixed_old',b&~o),('blend_broke_old',~b&o),
                ('blend_fixed_agmfa',b&~a),('blend_broke_agmfa',~b&a)]
    complement=pd.DataFrame([{'category':name,'count':int(mask.sum())} for name,mask in categories])
    overall=pd.DataFrame([{'model':name,**metrics(prediction)} for name,prediction in predictions.items()])
    per_class=[]
    for row in classes.itertuples():
        part=frame[frame.true_label==row.label]
        entry={'label':row.label,'class_name':row.class_name,'support':len(part)}
        entry.update({name+'_errors':int((~part[name+'_correct']).sum()) for name in MODELS})
        entry.update(fixed_old=int((~part.old_ensemble_correct&part.blend_50_50_correct).sum()),
                     broke_old=int((part.old_ensemble_correct&~part.blend_50_50_correct).sum()))
        per_class.append(entry)
    group_rows=[]
    for name,class_names in GROUPS.items():
        labels=classes.loc[classes.class_name.isin(class_names),'label']
        part=frame[frame.true_label.isin(labels)]
        group_rows.append({'group':name,'support':len(part),
                           **{model+'_errors':int((~part[model+'_correct']).sum()) for model in MODELS},
                           'fixed_old':int((~part.old_ensemble_correct&part.blend_50_50_correct).sum()),
                           'broke_old':int((part.old_ensemble_correct&~part.blend_50_50_correct).sum())})
    quality_rows=[]
    if val_dir is not None:
        directory=Path(val_dir).resolve()
        for image_id in frame.id:
            path=(directory/image_id).resolve()
            if not path.is_relative_to(directory):
                raise ValueError('Image ID escapes val directory')
            with Image.open(path) as source:
                gray=source.convert('L')
                array=np.asarray(gray,dtype=float)/255
                smooth=np.asarray(gray.filter(ImageFilter.GaussianBlur(1)),dtype=float)/255
            quality_rows.append({'id':image_id,'brightness':array.mean(),'contrast':array.std(),
                                 'detail':np.abs(array-smooth).mean()})
        frame=frame.merge(pd.DataFrame(quality_rows),on='id',validate='one_to_one')
    return {'predictions':predictions,'frame':frame,'metrics':overall,'complementarity':complement,
            'per_class':pd.DataFrame(per_class),'groups':pd.DataFrame(group_rows),'classes':classes}


def save_validation_report(result,root,run_id,agmfa_path,old_path,context):
    root=Path(root)
    if not run_id or Path(run_id).name!=run_id or any(char in run_id for char in ('/','\\',':')):
        raise ValueError('run_id must be a simple directory name')
    logs=root/'artifacts/logs'/run_id
    predictions=root/'artifacts/predictions'/run_id
    manifest={'rule':'0.5 AGMFA + 0.5 old ensemble','weights':[.5,.5],
              'inputs':{'agmfa':{'path':str(Path(agmfa_path).resolve()),'sha256':sha256(agmfa_path)},
                        'old_ensemble':{'path':str(Path(old_path).resolve()),'sha256':sha256(old_path)}},
              'context':context,'fitting':False,'weight_search':False,'test_evaluated':False}
    if logs.exists() or predictions.exists():
        saved=logs/'manifest.json'
        if not saved.is_file() or json.loads(saved.read_text(encoding='utf-8'))!=manifest:
            raise FileExistsError('Different/incomplete fusion run already exists; use a new run_id')
        if not (predictions/'val_probabilities.npz').is_file():
            raise FileExistsError('Incomplete fusion output; preserve it and use a new run_id')
        return {'logs':logs,'predictions':predictions}  # read-only reuse
    logs.mkdir(parents=True)
    predictions.mkdir(parents=True)
    for name in ('metrics','complementarity','per_class','groups','frame'):
        result[name].to_csv(logs/f'{name}.csv',index=False)
    save_probabilities(result['predictions']['blend_50_50'],predictions/'val_probabilities.npz')
    result['predictions']['blend_50_50'].table().to_csv(predictions/'val.csv',index=False)
    (logs/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return {'logs':logs,'predictions':predictions}


def case_gallery(result,subset,title,val_dir,limit=12):
    import math
    import matplotlib.pyplot as plt
    if limit<1:
        raise ValueError('Positive gallery limit required')
    names=result['classes'].set_index('label').class_name.to_dict()
    rows=[]
    for _,row in subset.head(limit).iterrows():
        caption=[f'True: {names[row.true_label]}']
        for model in MODELS:
            caption.append(f'{model}: {names[row[model+"_label"]]} {row[model+"_confidence"]:.0%}')
        rows.append((row.id,'\n'.join(caption)))
    if not rows:
        print(f'{title}: no images')
        return []
    directory=Path(val_dir).resolve()
    fig,axes=plt.subplots(math.ceil(len(rows)/3),3,figsize=(12,math.ceil(len(rows)/3)*5.2),
                          squeeze=False,layout='constrained')
    for ax in axes.flat:
        ax.axis('off')
    for ax,(image_id,caption) in zip(axes.flat,rows):
        path=(directory/image_id).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Image ID escapes val directory')
        with Image.open(path) as source:
            ax.imshow(source.convert('RGB'),interpolation='nearest')
        ax.set_title(f'{image_id}\n{caption}',fontsize=8)
    fig.suptitle(title,fontsize=12)
    plt.show()
    return [fig]


def export_test_fusion(agmfa_path,old_path,sample_path,classes_csv,root,run_id,
                       *,member_submissions,validation_manifest):
    """One candidate from train+val caches; no network inference, fitting or upload."""
    root=Path(root)
    if not run_id or Path(run_id).name!=run_id or any(char in run_id for char in ('/','\\',':')):
        raise ValueError('run_id must be a simple directory name')
    if len(member_submissions)!=2:
        raise ValueError('Provide exactly two member submissions for provenance checks')
    if not validation_manifest:
        raise ValueError('Provide the completed val report manifest')
    validation=json.loads(Path(validation_manifest).read_text(encoding='utf-8'))
    if validation['weights']!=[.5,.5] or validation.get('weight_search'):
        raise ValueError('Keep the predeclared weights, no search')
    a,o=load_probabilities(agmfa_path),load_probabilities(old_path)
    if a.true_labels is not None or o.true_labels is not None:
        raise ValueError('Test cache must not contain targets')
    sample=pd.read_csv(sample_path,dtype={'id':str})
    if len(sample)!=8299 or sample.id.isna().any() or sample.id.duplicated().any():
        raise ValueError('Require 8299 unique sample submission IDs')
    for prediction,path in zip((a,o),member_submissions):
        submitted=pd.read_csv(path,dtype={'id':str})
        if submitted.id.duplicated().any() or set(submitted.id)!=set(prediction.ids):
            raise ValueError('Member probability/submission IDs differ')
        expected=prediction.table().set_index('id').loc[submitted.id].label.to_numpy()
        np.testing.assert_array_equal(expected,submitted.label.to_numpy())
    classes=pd.read_csv(classes_csv)
    if set(a.class_labels)!=set(classes.label):
        raise ValueError('Unknown test classes')
    blend=ensemble_probabilities([a,o],[.5,.5])
    if len(blend.ids)!=8299 or set(blend.ids)!=set(sample.id):
        raise ValueError('Test/sample ID sets differ')
    indices=pd.Index(blend.ids).get_indexer(sample.id)
    blend.ids=blend.ids[indices]
    blend.probabilities=blend.probabilities[indices]
    blend.validate()
    log=root/'artifacts/logs'/run_id/'test_fusion.json'
    npz=root/'artifacts/predictions'/run_id/'test_probabilities.npz'
    csv=root/'artifacts/predictions'/run_id/'test.csv'
    submission=root/'submissions/submission_agmfa_old_ensemble_50_50.csv'
    if any(path.exists() for path in (log,npz,csv,submission)):
        raise FileExistsError('Fusion candidate exists; do not overwrite')
    if not log.parent.is_dir() or not npz.parent.is_dir():
        raise FileNotFoundError('First save validation report with this run_id')
    save_probabilities(blend,npz)
    blend.table().to_csv(csv,index=False)
    save_submission(blend.table(),sample_path,submission)
    exported=pd.read_csv(submission,dtype={'id':str})
    np.testing.assert_array_equal(exported.id.to_numpy(),sample.id.to_numpy())
    np.testing.assert_array_equal(exported.label.to_numpy(),blend.class_labels[blend.probabilities.argmax(1)])
    report={'rule':'0.5 AGMFA + 0.5 old ensemble (0.25 ResNet101 + 0.25 ResNet50 D4)',
            'weights':[.5,.5],'rows':8299,'fitting':False,'inference':False,'upload':False,
            'validation_manifest_sha256':sha256(validation_manifest),
            'sources':{str(agmfa_path):sha256(agmfa_path),str(old_path):sha256(old_path)},
            'submission':str(submission),'submission_sha256':sha256(submission)}
    log.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
