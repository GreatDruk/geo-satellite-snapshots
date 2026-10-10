"""One fixed 50/50 fusion of saved probabilities; no training or weight search."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from archive.src.training.predict import (load_probabilities,save_probabilities,ensemble_probabilities,save_submission)

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
