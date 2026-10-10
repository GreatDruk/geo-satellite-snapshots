from archive.src.utils import resolve_raw_dir
"""Deterministic full-frame/crop validation diagnostics; no training or test fitting."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from archive.src.data import ImageDataset, read_classes
from archive.src.experiment import PROJECT_ROOT, load_checkpoint
from archive.src.inference import file_hash, _metric
from archive.src.transforms import ImageTransform
from archive.src.training.predict import ProbabilityPredictions, load_probabilities, save_probabilities, tta_views, ensemble_probabilities
from archive.src.utils import get_device, set_seed

CROPS = ("top_left", "top_right", "center", "bottom_left", "bottom_right")
OUTPUTS = ("none", "flips", "d4") + CROPS


def crop_boxes(size, crop_size=160):
    width, height = size
    if not isinstance(crop_size, int) or not 0 < crop_size <= min(width, height):
        raise ValueError("crop_size must fit both image dimensions")
    right, bottom = width-crop_size, height-crop_size
    starts = [(0, 0), (right, 0), (right//2, bottom//2), (0, bottom), (right, bottom)]
    return [(x, y, x+crop_size, y+crop_size) for x,y in starts]


class LocalViews:
    """Crop original RGB pixels before resize/normalization; no augmentation."""
    def __init__(self, config, crop_size=160):
        self.transform = ImageTransform(config, training=False)
        self.crop_size = crop_size

    def __call__(self, image):
        images = [image] + [image.crop(box) for box in crop_boxes(image.size, self.crop_size)]
        return torch.stack([self.transform(view) for view in images])


def local_loader(raw, config, class_labels, split="val", crop_size=160, batch_size=4, num_workers=0, device="cpu"):
    mapping = {label: i for i,label in enumerate(class_labels)} if split == "val" else None
    dataset = ImageDataset(Path(raw)/f"{split}.csv", Path(raw)/split,
                           LocalViews(config, crop_size), mapping)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=num_workers, pin_memory=torch.device(device).type == "cuda")


def fuse_global_local(global_prediction, crops, method="mean", global_weight=.75):
    """Fixed probability fusion, aligning IDs/class columns; max is normalized."""
    if method not in {"mean", "max"} or not 0 <= global_weight <= 1 or len(crops) != 5:
        raise ValueError("Five crops, mean/max and a weight in [0,1] required")
    global_prediction.validate()
    # Reuse strict consistency validation (including GT), then explicitly align.
    ensemble_probabilities([global_prediction]+list(crops))
    aligned = [p.probabilities[pd.Index(p.ids).get_indexer(global_prediction.ids)][:,
               pd.Index(p.class_labels).get_indexer(global_prediction.class_labels)] for p in crops]
    local = np.stack(aligned).mean(0) if method == "mean" else np.stack(aligned).max(0)
    local = local / local.sum(1, keepdims=True)
    values = global_weight*global_prediction.probabilities + (1-global_weight)*local
    return ProbabilityPredictions(global_prediction.ids.copy(), global_prediction.class_labels.copy(),
                                  values.astype(np.float32), global_prediction.true_labels).validate()


def diagnostic_tables(predictions):
    full = predictions["none"]
    full.validate()
    truth = full.true_labels
    if truth is None:
        raise ValueError("Diagnostic requires validation labels")
    base = full.class_labels[full.probabilities.argmax(1)]
    rows = []
    for i,image_id in enumerate(full.ids):
        actual_index = int(np.flatnonzero(full.class_labels == truth[i])[0])
        ranks = [int(np.flatnonzero(np.argsort(-predictions[name].probabilities[i],kind="stable") == actual_index)[0])+1 for name in CROPS]
        crop_correct = [full.class_labels[predictions[name].probabilities[i].argmax()] == truth[i] for name in CROPS]
        full_rank = int(np.flatnonzero(np.argsort(-full.probabilities[i],kind="stable") == actual_index)[0])+1
        rows.append({"id":image_id,"true_label":truth[i],"full_label":base[i],"full_correct":bool(base[i]==truth[i]),
                     "full_true_rank":full_rank,"any_crop_true_top1":any(crop_correct),
                     "any_crop_true_top3":min(ranks)<=3,"new_crop_top3":full_rank>3 and min(ranks)<=3,
                     "wrong_crops":5-sum(crop_correct),
                     "confident_wrong_crops":sum(not ok and predictions[name].probabilities[i].max()>=.95 for name,ok in zip(CROPS,crop_correct))})
    return pd.DataFrame(rows)


@torch.inference_mode()
def local_probability_predictions(model, loader, device, class_labels, *, method="mean", global_weight=.75):
    """Replay full+five-crop fusion on sequential val/test, without fitting."""
    from torch.utils.data import SequentialSampler
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Sequential inference loader required")
    model.eval()
    parts={name:[] for name in ('none',)+CROPS}
    for views,_ in loader:
        if views.ndim!=5 or views.shape[1]!=6:
            raise ValueError("Expected full frame plus five crops")
        for i,name in enumerate(parts):
            logits=model(views[:,i].to(device).contiguous())
            if logits.shape!=(len(views),len(class_labels)):
                raise ValueError("Invalid classifier output")
            parts[name].append(logits.float().softmax(1).cpu().numpy())
    data=loader.dataset.data
    truth=data.label.to_numpy() if getattr(loader.dataset,'label_to_index',None) is not None else None
    predictions={name:ProbabilityPredictions(data.id.to_numpy(dtype=str),np.asarray(class_labels),
                 np.concatenate(values),truth).validate() for name,values in parts.items()}
    return fuse_global_local(predictions['none'],[predictions[n] for n in CROPS],method,global_weight)


@torch.inference_mode()
def run_local_diagnostic(checkpoint_path, run_id, *, project_root=PROJECT_ROOT, raw_dir="data/raw",
                         device="auto", batch_size=4, num_workers=0, crop_size=160,
                         cpu_threads=4, resume=False):
    """13 shared forwards/batch: 8 full orientations + 5 crops; resumable chunks.

    Resume validates checkpoint, CSVs, protocol, class/ID/target order. A complete
    run is read, never overwritten. Training journal and source artifacts untouched.
    """
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run_id")
    if batch_size < 1 or num_workers < 0 or cpu_threads < 1:
        raise ValueError("Invalid loader/thread settings")
    root = Path(project_root).resolve()
    raw = Path(raw_dir); raw = resolve_raw_dir(root, raw)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else root/checkpoint_path
    logs, outputs = root/'artifacts/logs'/run_id, root/'artifacts/predictions'/run_id
    metadata_path = logs/'metadata.json'
    protocol = {"version":1,"checkpoint_sha256":file_hash(checkpoint_path),
                "val_csv_sha256":file_hash(raw/'val.csv'),"classes_csv_sha256":file_hash(raw/'classes.csv'),
                "crop_size":crop_size,"crop_order":list(CROPS),"global_weight":.75,"batch_size":batch_size}
    previous = {}
    if logs.exists() or outputs.exists():
        if not resume or not metadata_path.is_file():
            raise FileExistsError("Choose a new run_id or resume an existing diagnostic")
        previous=json.loads(metadata_path.read_text(encoding='utf-8'))
        if previous['protocol'] != protocol:
            raise ValueError("Resume protocol/source mismatch")
        if previous['status']=='complete':
            for filename,expected in previous.get('probabilities_sha256',{}).items():
                if file_hash(outputs/filename)!=expected:
                    raise ValueError("Completed probability cache changed")
            return {"paths":{"logs":logs,"predictions":outputs},"comparison":pd.read_csv(logs/'comparison.csv'),
                    "diagnostic":pd.read_csv(logs/'crop_diagnostic.csv'),"selection":json.loads((logs/'selection.json').read_text(encoding='utf-8'))}
    else:
        logs.mkdir(parents=True); outputs.mkdir(parents=True)
    chunks=outputs/'chunks'; chunks.mkdir(exist_ok=True)
    selected_device=get_device(device)
    if selected_device.type=='cpu': torch.set_num_threads(cpu_threads)
    set_seed(42, deterministic=True, warn_only=True)
    metadata={"kind":"global_local_validation","status":"running","protocol":protocol,
              "device":str(selected_device),"torch":str(torch.__version__),"test_read":False,"training":False,
              "seed":42,"deterministic":True,"deterministic_warn_only":True,
              "attempts":previous.get('attempts',[])+[{'device':str(selected_device),'torch':str(torch.__version__)}]}
    def update(): metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    def emit(message):
        print(message,flush=True)
        with (logs/'inference.log').open('a',encoding='utf-8') as f: f.write(message+'\n')
    update(); started=time.perf_counter()
    try:
        model, checkpoint=load_checkpoint(checkpoint_path,selected_device,inference_only=True)
        classes=read_classes(raw)
        labels=np.asarray(checkpoint['class_labels'])
        if set(labels)!=set(classes.label): raise ValueError("Checkpoint class mismatch")
        loader=local_loader(raw,checkpoint['config']['data'],labels,crop_size=crop_size,
                            batch_size=batch_size,num_workers=num_workers,device=selected_device)
        data=loader.dataset.data
        parts={name:[] for name in OUTPUTS}
        for batch_index,(views,targets) in enumerate(loader):
            start=batch_index*batch_size; ids=data.id.iloc[start:start+len(views)].to_numpy(dtype=str)
            truth=data.label.iloc[start:start+len(views)].to_numpy()
            cache=chunks/f'{batch_index:05d}.npz'
            if cache.exists():
                with np.load(cache,allow_pickle=False) as arrays:
                    if not np.array_equal(arrays['ids'],ids) or not np.array_equal(arrays['class_labels'],labels) or not np.array_equal(arrays['true_labels'],truth):
                        raise ValueError("Invalid resumed chunk order/targets")
                    for name in OUTPUTS:
                        part=ProbabilityPredictions(ids,labels,arrays[name],truth).validate()
                        parts[name].append(part.probabilities)
            else:
                values={}; full=views[:,0].to(selected_device,non_blocking=selected_device.type=='cuda')
                orientation=[model(v.contiguous()).float().softmax(1) for v in tta_views(full,'d4')]
                values['none']=orientation[0].cpu().numpy()
                values['flips']=torch.stack([orientation[i] for i in [0,1,4,5]]).mean(0).cpu().numpy()
                values['d4']=torch.stack(orientation).mean(0).cpu().numpy()
                for i,name in enumerate(CROPS,1):
                    values[name]=model(views[:,i].to(selected_device,non_blocking=selected_device.type=='cuda').contiguous()).float().softmax(1).cpu().numpy()
                for name in OUTPUTS:
                    ProbabilityPredictions(ids,labels,values[name],truth).validate()
                    parts[name].append(values[name])
                temporary=cache.with_suffix('.tmp')
                with temporary.open('wb') as f: np.savez_compressed(f,ids=ids,class_labels=labels,true_labels=truth,**values)
                temporary.rename(cache)
            if batch_index==0 or (batch_index+1)%5==0 or batch_index+1==len(loader):
                emit(f'Batch {batch_index+1}/{len(loader)}; elapsed {(time.perf_counter()-started)/60:.2f} min')
        predictions={name:ProbabilityPredictions(data.id.to_numpy(dtype=str),labels,np.concatenate(values),data.label.to_numpy()).validate() for name,values in parts.items()}
        predictions['global_local_mean']=fuse_global_local(predictions['none'],[predictions[n] for n in CROPS],'mean')
        predictions['global_local_max']=fuse_global_local(predictions['none'],[predictions[n] for n in CROPS],'max')
        rows=[]
        base_ok=predictions['none'].table().label.to_numpy()==data.label.to_numpy()
        for name,prediction in predictions.items():
            filename=outputs/f'{name}.npz'
            if filename.exists():
                old=load_probabilities(filename)
                if not np.array_equal(old.ids,prediction.ids) or not np.array_equal(old.probabilities,prediction.probabilities):
                    raise ValueError("Existing result differs from resumed computation")
            else: save_probabilities(prediction,filename)
            prediction.table(include_scores=True).to_csv(outputs/f'{name}.csv',index=False)
            ok=prediction.table().label.to_numpy()==data.label.to_numpy()
            rows.append({'recipe':name,**_metric(prediction),'fixes':int((~base_ok&ok).sum()),'breaks':int((base_ok&~ok).sum()),
                         'views':8 if name=='d4' else 4 if name=='flips' else 6 if name.startswith('global_local') else 1,
                         'candidate':name not in CROPS})
        comparison=pd.DataFrame(rows).sort_values(['val_accuracy','views','val_nll'],ascending=[False,True,True]).reset_index(drop=True)
        comparison.to_csv(logs/'comparison.csv',index=False)
        diagnostic=diagnostic_tables(predictions);diagnostic.to_csv(logs/'crop_diagnostic.csv',index=False)
        best=comparison.loc[comparison.candidate].iloc[0]
        portable=str(checkpoint_path.relative_to(root)).replace('\\','/') if checkpoint_path.is_relative_to(root) else str(checkpoint_path)
        selection={'format_version':2 if best.recipe.startswith('global_local') else 1,'selected_on':'val','recipe':best.recipe,
                   'val_accuracy':float(best.val_accuracy),'val_nll':float(best.val_nll),'val_csv_sha256':protocol['val_csv_sha256'],
                   'class_labels':labels.tolist(),'members':[{'checkpoint':portable,'sha256':protocol['checkpoint_sha256'],
                    'name':'resnet101','tta':best.recipe if best.recipe in ('none','flips','d4') else 'none','weight':1.0}]}
        if selection['format_version']==2:
            selection.update(method='global_local',crop_size=crop_size,global_weight=.75,local_pool=best.recipe.rsplit('_',1)[1])
        (logs/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        wrong=diagnostic.loc[~diagnostic.full_correct]
        metadata.update(status='complete',elapsed_min=(time.perf_counter()-started)/60,
                        crop_recovered_errors=int(wrong.any_crop_true_top1.sum()),
                        probabilities_sha256={f'{name}.npz':file_hash(outputs/f'{name}.npz') for name in predictions})
        emit(comparison.to_string(index=False));emit(f'Oracle crop recovery: {metadata["crop_recovered_errors"]}/{len(wrong)} errors; NOT ensemble quality')
        del model,checkpoint,loader
    except BaseException as error:
        metadata.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally: update()
    return {'paths':{'logs':logs,'predictions':outputs},'comparison':comparison,'diagnostic':diagnostic,'selection':selection}


def analyze_saved_local(run_id, *, project_root=PROJECT_ROOT, raw_dir="data/raw", reference_run="quality_inference_001"):
    """Compare two predetermined ResNet50 combinations from verified cached arrays."""
    root=Path(project_root).resolve()
    logs=root/'artifacts/logs'/run_id;outputs=root/'artifacts/predictions'/run_id
    metadata=json.loads((logs/'metadata.json').read_text(encoding='utf-8'))
    if metadata['status']!='complete': raise ValueError("Wait for complete diagnostic")
    raw=Path(raw_dir);raw=raw if raw.is_absolute() else root/raw
    if file_hash(raw/'val.csv')!=metadata['protocol']['val_csv_sha256'] or file_hash(raw/'classes.csv')!=metadata['protocol']['classes_csv_sha256']:
        raise ValueError("Validation/classes changed since diagnostic")
    for filename,expected in metadata['probabilities_sha256'].items():
        if file_hash(outputs/filename)!=expected:raise ValueError("Diagnostic probability cache changed")
    comparison=pd.read_csv(logs/'comparison.csv')
    base=load_probabilities(outputs/'none.npz')
    base_ok=base.table().label.to_numpy()==base.true_labels
    original_selection=json.loads((logs/'selection.json').read_text(encoding='utf-8'))
    selections={name:{**original_selection,'format_version':1,'recipe':name,
                 'members':[{**original_selection['members'][0],'tta':name}]}
                for name in ['none','flips','d4']}
    for method in ['mean','max']:
        name='global_local_'+method
        selections[name]={**original_selection,'format_version':2,'recipe':name,'method':'global_local',
                          'local_pool':method,'global_weight':.75,'crop_size':metadata['protocol']['crop_size'],
                          'members':[{**original_selection['members'][0],'tta':'none'}]}
    reference_logs=root/'artifacts/logs'/reference_run
    if (reference_logs/'metadata.json').is_file():
        cache=json.loads((reference_logs/'metadata.json').read_text(encoding='utf-8'))
        if cache['status']!='complete' or any(cache[k]!=metadata['protocol'][k] for k in ['val_csv_sha256','classes_csv_sha256']):
            raise ValueError("Reference data/protocol mismatch")
        spec=cache['sources']['stronger']
        checkpoint=Path(spec['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else root/checkpoint
        if file_hash(checkpoint)!=spec['sha256']:raise ValueError("Reference checkpoint changed")
        new_rows=[]
        for new_mode,old_mode in [('none','d4'),('flips','flips')]:
            filename=f'stronger__{old_mode}.npz'
            source=root/'artifacts/predictions'/reference_run/filename
            if file_hash(source)!=cache['probabilities_sha256'][filename]:raise ValueError("Reference probabilities changed")
            result=ensemble_probabilities([load_probabilities(outputs/f'{new_mode}.npz'),load_probabilities(source)])
            name=f'resnet101_{new_mode}_plus_stronger_{old_mode}'
            destination=outputs/f'{name}.npz'
            if destination.exists():
                saved=load_probabilities(destination)
                if not np.array_equal(saved.probabilities,result.probabilities):raise ValueError("Derived ensemble changed")
            else:save_probabilities(result,destination)
            result.table(include_scores=True).to_csv(outputs/f'{name}.csv',index=False)
            correct=result.table().label.to_numpy()==result.true_labels
            new_rows.append({'recipe':name,**_metric(result),'fixes':int((~base_ok&correct).sum()),
                             'breaks':int((base_ok&~correct).sum()),'views':9 if new_mode=='none' else 8,'candidate':True})
            selections[name]={**original_selection,'format_version':1,'recipe':name,
                              'members':[{**original_selection['members'][0],'tta':new_mode,'weight':.5},
                                         {**spec,'name':'stronger','tta':old_mode,'weight':.5}]}
        comparison=pd.concat([comparison,pd.DataFrame(new_rows)],ignore_index=True)
    comparison=comparison.sort_values(['val_accuracy','views','val_nll'],ascending=[False,True,True]).reset_index(drop=True)
    comparison.to_csv(logs/'final_comparison.csv',index=False)
    best=comparison.loc[comparison.candidate].iloc[0]
    selection={**selections[best.recipe],'val_accuracy':float(best.val_accuracy),'val_nll':float(best.val_nll)}
    (logs/'final_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    diagnostic=pd.read_csv(logs/'crop_diagnostic.csv')
    wrong=diagnostic.loc[~diagnostic.full_correct]
    good=diagnostic.loc[diagnostic.full_correct]
    summary={'full_errors':len(wrong),'crop_oracle_recovered':int(wrong.any_crop_true_top1.sum()),
             'new_crop_top3_errors':int(wrong.new_crop_top3.sum()),
             'correct_images_with_any_wrong_crop':int((good.wrong_crops>0).sum()),
             'correct_images_with_confident_wrong_crop':int((good.confident_wrong_crops>0).sum()),
             'best_recipe':str(best.recipe),'best_correct':int(best.correct),'best_accuracy':float(best.val_accuracy),
             'crop_oracle_is_not_model_quality':True}
    classes=read_classes(raw)
    names=classes.set_index('label').class_name.to_dict()
    summary['recovered_by_class']={names[label]:int(count) for label,count in wrong.loc[wrong.any_crop_true_top1].groupby('true_label').size().items()}
    from archive.src.error_analysis import class_metrics, confusion_pairs
    selected=load_probabilities(outputs/f'{best.recipe}.npz').table(include_scores=True).set_index('id').rename(columns={'label':'prediction'})
    class_metrics(selected,'prediction',classes).to_csv(logs/'selected_class_metrics.csv',index=False)
    confusion_pairs(selected,'prediction',classes).to_csv(logs/'selected_confusions.csv',index=False)
    source_exp=Path(original_selection['members'][0]['checkpoint']).parent.name
    original=root/'artifacts/predictions'/source_exp/'val_probabilities.npz'
    if original.exists():
        saved=load_probabilities(original)
        ensemble_probabilities([base,saved]) # verifies ID/classes/targets
        rows=pd.Index(saved.ids).get_indexer(base.ids)
        saved_predictions=saved.class_labels[saved.probabilities[rows].argmax(1)]
        summary['reproduced_original_argmax']=int((base.class_labels[base.probabilities.argmax(1)]==saved_predictions).sum())
    (logs/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return {'comparison':comparison,'summary':summary,'selection':selection,'diagnostic':diagnostic,'paths':{'logs':logs,'predictions':outputs}}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--device',default='auto');parser.add_argument('--raw-dir',default='data/raw')
    parser.add_argument('--batch-size',type=int,default=4);parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run_local_diagnostic(args.checkpoint,args.run_id,raw_dir=args.raw_dir,device=args.device,
                         batch_size=args.batch_size,cpu_threads=args.cpu_threads,resume=args.resume)
