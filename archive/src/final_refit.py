from archive.src.utils import resolve_raw_dir
"""Explicit final train+val refit, fixed epochs, no held-out metric/selection.

Separate from run_experiment so ordinary train/val experiments remain intact.
Only the final epoch is the deployable checkpoint. Test inference is opt-in.
"""
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from archive.src.data import ImageDataset, read_classes, make_loader, load_test
from archive.src.experiment import PROJECT_ROOT, load_config, make_optimizer
from archive.src.inference import file_hash
from archive.src.models import build_model
from archive.src.transforms import ImageTransform
from archive.src.training.train import train_model
from archive.src.training.schedulers import make_scheduler
from archive.src.training.predict import probability_predictions, save_probabilities, save_submission
from archive.src.utils import set_seed, get_device, validate_journal, append_experiment


def refit_loader(raw, data_config, seed):
    """Both labeled splits get the fixed train transform; raw remains immutable."""
    raw = Path(raw)
    classes = read_classes(raw)
    mapping = {label:i for i,label in enumerate(classes.label)}
    transform = ImageTransform(data_config, training=True)
    splits = [ImageDataset(raw/f'{split}.csv', raw/split, transform, mapping) for split in ('train','val')]
    dataset = ConcatDataset(splits)
    loader = make_loader(dataset, data_config, seed, shuffle=True)
    return loader, classes, {name:len(ds) for name,ds in zip(('train','val'),splits)}


def run_final_refit(config, exp_id=None, *, project_root=PROJECT_ROOT, model_factory=None,
                    show_plots=True, predict_test=False):
    """Use only with an explicitly finalized recipe and authorized train+val fitting."""
    root = Path(project_root).resolve()
    config = load_config(config, root)
    if config.get('final_refit',{}).get('training_splits') != ['train','val']:
        raise ValueError('Explicit final_refit.training_splits=[train,val] required')
    exp_id = exp_id or config.get('exp_id')
    if not isinstance(exp_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',exp_id):
        raise ValueError('Invalid experiment ID')
    config['exp_id'] = exp_id
    training = config['training']
    epochs = training['epochs']
    if not isinstance(epochs,int) or epochs<1:
        raise ValueError('Positive fixed epoch count required')
    if any(k in training for k in ('initialize_from','hard_mining','batch_augmentation','loss')):
        raise ValueError('Final baseline refit supports fixed ordinary CE from ImageNet only')
    schedule = training.get('scheduler')
    if schedule and schedule['name'].lower() in {'plateau','reducelronplateau'}:
        raise ValueError('Validation-monitored schedules are forbidden during final refit')
    if config['model'].get('warmup_epochs',0) >= epochs:
        raise ValueError('Head warmup must be shorter than training')
    journal = (root.parent if root.name == 'archive' else root)/'experiments.csv'
    validate_journal(journal, exp_id)
    paths = {kind:root/f'artifacts/{kind}'/exp_id for kind in ('models','logs','predictions')}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('Use a new final refit experiment ID')
    submission = root/'submissions'/f'{exp_id}.csv'
    if predict_test and submission.exists():
        raise FileExistsError(submission)
    set_seed(config['seed'], training.get('deterministic',True), training.get('deterministic_warn_only',False))
    device = get_device(training.get('device','auto'))
    if device.type == 'cpu':
        config['data']['pin_memory'] = False
        if os.name == 'nt': config['data']['num_workers'] = 0
    raw = Path(config['data'].get('raw_dir','data/raw'))
    raw = resolve_raw_dir(root, raw)
    loader, classes, split_counts = refit_loader(raw,config['data'],config['seed'])
    model = build_model(config['model'],config['data']['image_size'],len(classes),model_factory).to(device)
    optimizer = make_optimizer(model, training['optimizer'])
    scheduler, monitor = make_scheduler(optimizer, schedule, epochs)
    if monitor is not None: raise ValueError('No validation scheduler monitor permitted')
    for path in paths.values(): path.mkdir(parents=True)
    (paths['logs']/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    metadata = {'kind':'final_train_val_refit', 'status':'running', 'training_splits':split_counts,
                'held_out_validation':False, 'checkpoint_selection':'fixed final epoch',
                'test_used_for_fitting':False, 'device':str(device), 'seed':config['seed'],
                'torch':str(torch.__version__), 'sources':{name:file_hash(raw/name)
                       for name in ('train.csv','val.csv','classes.csv')}}
    metadata_path = paths['logs']/'metadata.json'
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    def emit(line):
        print(line,flush=True)
        with (paths['logs']/'training.log').open('a',encoding='utf-8') as stream: stream.write(line+'\n')
    started = time.perf_counter(); rows=[]
    try:
        for epoch in range(1,epochs+1):
            epoch_started = time.perf_counter()
            if hasattr(model,'set_epoch'): model.set_epoch(epoch)
            rates = {g.get('name',str(i)):g['lr'] for i,g in enumerate(optimizer.param_groups)}
            loss,accuracy = train_model(model,loader,torch.nn.CrossEntropyLoss(),optimizer,device)
            if scheduler is not None: scheduler.step()
            rows.append({'epoch':epoch,'train_loss':loss,'train_accuracy':accuracy,
                         'learning_rates':json.dumps(rates),'epoch_time_sec':time.perf_counter()-epoch_started,
                         'train_samples':len(loader.dataset)})
            pd.DataFrame(rows).to_csv(paths['logs']/'history.csv',index=False)
            if epoch==1 or epoch%training.get('log_every',1)==0 or epoch==epochs:
                emit(f'Final refit {epoch}/{epochs}: train loss={loss:.4f} acc={accuracy:.4f}; lr={rates}; no held-out val')
        model.eval()
        checkpoint = {'model_state_dict':model.state_dict(),'epoch':epochs,'config':config,
                      'class_labels':classes.label.tolist(),'class_names':classes.class_name.tolist(),
                      'val_accuracy':None,'final_refit':True,'training_splits':['train','val']}
        torch.save(checkpoint,paths['models']/'final.pt')
        elapsed=(time.perf_counter()-started)/60
        # Train-only curves: intentionally no synthetic validation curve.
        import matplotlib.pyplot as plt
        history=pd.DataFrame(rows)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,column in zip(axes,('train_loss','train_accuracy')):
            ax.plot(history.epoch,history[column]); ax.set(xlabel='Epoch',ylabel=column); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(paths['logs']/'learning_curves.png',dpi=150)
        if show_plots: plt.show()
        plt.close(fig)
        if predict_test:
            test_loader=load_test(raw,config['data'],config['seed'])
            p=probability_predictions(model,test_loader,device,classes.label.tolist())
            save_probabilities(p,paths['predictions']/'test_probabilities.npz')
            p.table().to_csv(paths['predictions']/'test.csv',index=False)
            save_submission(p.table(),raw/'sample_submission_example.csv',submission)
        summary={'epochs':epochs,'train_time_min':elapsed,'val_accuracy':None,'checkpoint':'final.pt'}
        (paths['logs']/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        append_experiment(journal, {'exp_id':exp_id,'model':config['model']['name'],
            'image_size':json.dumps(config['data']['image_size']),
            'augmentation':json.dumps(config['data'].get('augmentation',{}),sort_keys=True),
            'pretrained':config['model'].get('pretrained',False),'val_accuracy':'',
            'train_time_min':elapsed,'seed':config['seed'],'submission_id':exp_id if predict_test else '',
            'public_score':'','notes':config.get('notes','')+'; FINAL train+val; no held-out val accuracy'})
        metadata.update(status='complete',checkpoint_sha256=file_hash(paths['models']/'final.pt'))
        metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    except BaseException as error:
        metadata.update(status='failed',error=f'{type(error).__name__}: {error}')
        metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        raise
    return {'model':model,'history':history,'paths':paths,'summary':summary}
