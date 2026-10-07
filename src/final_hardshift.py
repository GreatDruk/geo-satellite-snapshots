"""Two fixed-epoch final AGMFA runs, with origin-aware target-domain training.

Training never reads test. Test inference is a separate explicit operation.
Previous Hard Target Shift experiments and baseline refits remain unchanged.
"""
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from src.data import ImageDataset, make_loader, read_classes
from src.experiment import load_config, load_checkpoint, make_optimizer
from src.hard_target_shift import HardTargetDataset, PROFILES
from src.hard_target_training import validate_recipe
from src.models import build_model
from src.training.evaluate import evaluate_model
from src.training.predict import probability_predictions, save_probabilities, save_submission, load_probabilities
from src.training.schedulers import make_scheduler
from src.transforms import ImageTransform, apply_d4
from src.utils import get_device, set_seed, validate_journal, append_experiment, seed_worker


class TargetDomainDataset(ImageDataset):
    """Former val: shared epoch, uniform D4 ONLY, no synthetic corruptions."""
    def __init__(self, raw, data, mapping, seed):
        super().__init__(Path(raw)/'val.csv', Path(raw)/'val', label_to_index=mapping)
        self.tensor = ImageTransform(data, training=False)
        self.seed = seed
        self._epoch = torch.tensor(1, dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch):
        self._epoch.fill_(epoch)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(self._epoch), index, 1]))
        image = image.resize((256,256), Image.Resampling.BILINEAR)
        return self.tensor(apply_d4(image, int(rng.integers(8)))), label


class OriginDataset(Dataset):
    def __init__(self, source, target=None):
        self.source, self.target = source, target

    def __len__(self):
        return len(self.source) + (len(self.target) if self.target is not None else 0)

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)
        if self.target is not None:
            self.target.set_epoch(epoch)

    def __getitem__(self, index):
        if index < len(self.source):
            clean, hard, label, profile, local_index = self.source[index]
            return {'origin':'source', 'clean':clean, 'hard':hard, 'label':label,
                    'profile':profile, 'index':local_index}
        local_index = index-len(self.source)
        image, label = self.target[local_index]
        return {'origin':'target', 'clean':image, 'label':label, 'profile':-1, 'index':local_index}


def origin_collate(items):
    source = [item for item in items if item['origin']=='source']
    target = [item for item in items if item['origin']=='target']
    # No placeholder hard image for target: it receives exactly ONE forward.
    result = {}
    for name, rows in (('source',source), ('target',target)):
        result[name+'_labels'] = torch.tensor([r['label'] for r in rows],dtype=torch.long)
        result[name+'_indices'] = torch.tensor([r['index'] for r in rows],dtype=torch.long)
        result[name+'_images'] = torch.stack([r['clean'] for r in rows]) if rows else None
    result['source_hard'] = torch.stack([r['hard'] for r in source]) if source else None
    result['source_profiles'] = torch.tensor([r['profile'] for r in source],dtype=torch.long)
    return result


def source_object_losses(clean, hard, labels):
    clean_ce, hard_ce = F.cross_entropy(clean,labels,reduction='none'), F.cross_entropy(hard,labels,reduction='none')
    log_p, log_q = F.log_softmax(clean.float(),1), F.log_softmax(hard.float(),1)
    log_m = torch.logaddexp(log_p,log_q)-math.log(2)
    js = .5*((log_p.exp()*(log_p-log_m)).sum(1)+(log_q.exp()*(log_q-log_m)).sum(1))
    return .5*clean_ce+.5*hard_ce+3*js, {'clean_ce':clean_ce,'hard_ce':hard_ce,'js_loss':js}


def mixed_objective(clean, hard, source_labels, target_logits, target_labels, target_weight=2.):
    """Exact weighted per-object numerator, with original-object denominator."""
    n_source, n_target = len(source_labels), len(target_labels)
    if not n_source and not n_target:
        raise ValueError('Empty batch')
    terms = {}
    numerator = None
    if n_source:
        losses, terms = source_object_losses(clean,hard,source_labels)
        numerator = losses.sum()
    if n_target:
        terms['target_ce'] = F.cross_entropy(target_logits,target_labels,reduction='none')
        target_sum = target_weight*terms['target_ce'].sum()
        numerator = target_sum if numerator is None else numerator+target_sum
    mass = n_source+target_weight*n_target
    return numerator/mass, terms, mass


def train_origin_epoch(model, loader, optimizer, device, target_weight=2.):
    model.train()
    totals = dict(clean_ce=0.,hard_ce=0.,js_loss=0.,clean_accuracy=0.,hard_accuracy=0.,target_ce=0.,target_accuracy=0.)
    source_seen = np.zeros(len(loader.dataset.source),dtype=int)
    target_seen = np.zeros(len(loader.dataset.target) if loader.dataset.target is not None else 0,dtype=int)
    profile_counts = np.zeros(6,dtype=int)
    numerator_sum, mass_sum, forward_count = 0.,0.,0
    for batch in loader:
        ns, nt = len(batch['source_labels']),len(batch['target_labels'])
        non_blocking = device.type=='cuda' and loader.pin_memory
        images = [batch[key] for key in ('source_images','source_hard','target_images') if batch[key] is not None]
        inputs = torch.cat(images).to(device,non_blocking=non_blocking)
        sy = batch['source_labels'].to(device,non_blocking=non_blocking)
        ty = batch['target_labels'].to(device,non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        clean,hard,target = logits[:ns],logits[ns:2*ns],logits[2*ns:]
        loss,terms,mass = mixed_objective(clean,hard,sy,target,ty,target_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite final hardshift loss')
        loss.backward()
        optimizer.step()
        numerator_sum += loss.item()*mass
        mass_sum += mass
        forward_count += len(inputs)
        for key,value in terms.items():
            totals[key] += value.detach().sum().item()
        if ns:
            totals['clean_accuracy'] += (clean.argmax(1)==sy).sum().item()
            totals['hard_accuracy'] += (hard.argmax(1)==sy).sum().item()
            np.add.at(source_seen,batch['source_indices'].numpy(),1)
            profile_counts += np.bincount(batch['source_profiles'].numpy(),minlength=6)
        if nt:
            totals['target_accuracy'] += (target.argmax(1)==ty).sum().item()
            np.add.at(target_seen,batch['target_indices'].numpy(),1)
    if not len(source_seen) or not (source_seen==1).all() or not (target_seen==1).all():
        raise AssertionError('Every source/target object must appear exactly once per epoch')
    source_count,target_count = len(source_seen),len(target_seen)
    assert forward_count==2*source_count+target_count
    metrics = {key:totals[key]/source_count for key in ('clean_ce','hard_ce','js_loss','clean_accuracy','hard_accuracy')}
    metrics.update(train_loss=numerator_sum/mass_sum,source_samples=source_count,target_samples=target_count,
                   forward_images=forward_count,target_ce=totals['target_ce']/target_count if target_count else None,
                   target_accuracy=totals['target_accuracy']/target_count if target_count else None,
                   weighted_target_fraction=target_weight*target_count/(source_count+target_weight*target_count))
    return metrics,profile_counts


def validate_final_config(cfg):
    validate_recipe(cfg)
    final = cfg['final_hardshift']
    if final['training_splits'] not in (['train'],['train','val']) or final['target_weight']!=2.:
        raise ValueError('Require train-only or train+val with target weight2')
    if cfg['training']['epochs']!=22 or cfg['training']['scheduler']!={'name':'cosine','T_max':22,'eta_min':1e-6}:
        raise ValueError('Fixed 22 epochs and cosine T_max22 required')
    if cfg['training'].get('initialize_from') or cfg['seed']!=42:
        raise ValueError('Fresh ImageNet initialization and seed42 required')
    model = cfg['model']
    if model['name']!='resnet50_agmfa' or model['weights']!='IMAGENET1K_V2' or not model['pretrained']:
        raise ValueError('Corrected AGMFA with ImageNet V2 required')
    if model['warmup_epochs']!=3 or model['unfreeze_blocks']!=['layer3','layer4']:
        raise ValueError('Use warmup3 and partial layer3/layer4')
    if cfg['training']['optimizer']!={'name':'adamw','lr':5e-4,'backbone_lr':5e-5,'head_lr':5e-4,'weight_decay':.001}:
        raise ValueError('Keep the requested optimizer/LR')
    if final['checkpoint_selection']!='fixed_last_epoch':
        raise ValueError('Validation selection is forbidden')


def _raw(root,data):
    raw = Path(data['raw_dir'])
    return raw if raw.is_absolute() else root/raw


def run_final_hardshift(config, *, project_root=None, show_plots=True, model_factory=None):
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    cfg = load_config(config,root)
    validate_final_config(cfg)
    exp_id = cfg['exp_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',exp_id):
        raise ValueError('Invalid exp_id')
    validate_journal(root/'experiments.csv',exp_id)
    paths = {kind:root/'artifacts'/kind/exp_id for kind in ('models','logs','predictions')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Do not overwrite a final run; choose a fresh ID')
    train_val = cfg['final_hardshift']['training_splits']==['train','val']
    data,training = cfg['data'],cfg['training']
    set_seed(cfg['seed'],training['deterministic'],training['deterministic_warn_only'])
    device = get_device(training['device'])
    if device.type!='cuda':
        data['pin_memory']=False
    if os.name=='nt':
        data['num_workers']=0
    raw = _raw(root,data)
    classes = read_classes(raw)
    mapping = {label:index for index,label in enumerate(classes.label)}
    source = HardTargetDataset(raw/'train.csv',raw/'train',data,mapping,cfg['seed'])
    target = TargetDomainDataset(raw,data,mapping,cfg['seed']) if train_val else None
    expected = cfg['final_hardshift']['expected_counts']
    if len(source)!=expected['train'] or (train_val and len(target)!=expected['val']):
        raise ValueError('Unexpected source/target split sizes')
    dataset = OriginDataset(source,target)
    options = {'persistent_workers':data.get('persistent_workers',False),'prefetch_factor':data.get('prefetch_factor',2)} if data['num_workers'] else {}
    loader = DataLoader(dataset,batch_size=data['batch_size'],shuffle=True,num_workers=data['num_workers'],
                        pin_memory=data['pin_memory'],collate_fn=origin_collate,worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(cfg['seed']),**options)
    # Only train-only constructs a validation loader. Refit has no held-out val.
    val_loader = None
    if not train_val:
        val = ImageDataset(raw/'val.csv',raw/'val',ImageTransform(data,False),mapping)
        if len(val)!=expected['val']:
            raise ValueError('Unexpected reporting val size')
        val_loader = make_loader(val,data,cfg['seed'])
    model = build_model(cfg['model'],256,len(classes),model_factory,initialize_pretrained=True).to(device)
    optimizer = make_optimizer(model,training['optimizer'])
    scheduler,monitor = make_scheduler(optimizer,training['scheduler'],22)
    assert monitor is None and not optimizer.state
    for path in paths.values():
        path.mkdir(parents=True,exist_ok=False)
    metadata = {'status':'running','kind':'final_agmfa_hardshift','fresh_imagenet_initialization':True,
                'device':str(device),'seed':cfg['seed'],'torch':str(torch.__version__),
                'training_splits':cfg['final_hardshift']['training_splits'],'source_samples':len(source),
                'target_samples':len(target) if target is not None else 0,'target_weight':2.,
                'target_transforms':'D4 only; no hard profile','checkpoint_selection':'fixed_last_epoch',
                'fixed_epoch':22,'held_out_validation':not train_val,'test_read':False,'test_tta':'none'}
    (paths['logs']/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (paths['logs']/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    rows,counts_rows = [],[]
    started = time.perf_counter()
    def emit(message):
        print(message,flush=True)
        with (paths['logs']/'training.log').open('a',encoding='utf-8') as stream:
            stream.write(message+'\n')
    try:
        for epoch in range(1,23):
            epoch_start = time.perf_counter()
            dataset.set_epoch(epoch)
            loader.generator.manual_seed(cfg['seed']+epoch)
            model.set_epoch(epoch)
            rates = {g['name']:g['lr'] for g in optimizer.param_groups}
            metrics,counts = train_origin_epoch(model,loader,optimizer,device,2.)
            np.testing.assert_array_equal(counts,np.bincount([(i+epoch)%6 for i in range(len(source))],minlength=6))
            if val_loader is not None:
                metrics['val_loss'],metrics['val_accuracy']=evaluate_model(model,val_loader,torch.nn.CrossEntropyLoss(),device)
            scheduler.step()
            rows.append({'epoch':epoch,**metrics,'training_mode':model.active_mode,'learning_rates':json.dumps(rates),
                         'next_learning_rates':json.dumps({g['name']:g['lr'] for g in optimizer.param_groups}),
                         'epoch_time_sec':time.perf_counter()-epoch_start})
            counts_rows.extend({'epoch':epoch,'profile':name,'count':int(count)} for name,count in zip(PROFILES,counts))
            pd.DataFrame(rows).to_csv(paths['logs']/'history.csv',index=False)
            pd.DataFrame(counts_rows).to_csv(paths['logs']/'profile_counts.csv',index=False)
            if epoch==1 or epoch%training['log_every']==0 or epoch==22:
                emit(f'Epoch {epoch}/22 | source clean={metrics["clean_accuracy"]:.4f} hard={metrics["hard_accuracy"]:.4f} '
                     f'CE={metrics["clean_ce"]:.4f}/{metrics["hard_ce"]:.4f} JS={metrics["js_loss"]:.4f} '
                     f'| target CE={metrics["target_ce"]} | val={metrics.get("val_accuracy", "none")} '
                     f'| LR={rates} | selection=fixed last epoch')
        model.eval()
        checkpoint = {'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                      'scheduler_state_dict':scheduler.state_dict(),'epoch':22,'config':cfg,
                      'class_labels':classes.label.tolist(),'class_names':classes.class_name.tolist(),
                      'val_accuracy':rows[-1].get('val_accuracy'),'val_loss':rows[-1].get('val_loss'),
                      'training_splits':cfg['final_hardshift']['training_splits'],'final_refit':train_val,
                      'checkpoint_selection':'fixed_last_epoch'}
        torch.save(checkpoint,paths['models']/'final.pt')
        shutil.copyfile(paths['models']/'final.pt',paths['models']/'last.pt')
        if val_loader is not None:
            probabilities = probability_predictions(model,val_loader,device,classes.label.tolist(),tta='none')
            save_probabilities(probabilities,paths['predictions']/'val_probabilities.npz')
            probabilities.table().to_csv(paths['predictions']/'val.csv',index=False)
        elapsed = (time.perf_counter()-started)/60
        summary = {'epochs':22,'checkpoint':'final.pt' if train_val else 'last.pt','selection':'fixed_last_epoch',
                   'val_accuracy':rows[-1].get('val_accuracy'),'val_loss':rows[-1].get('val_loss'),'train_time_min':elapsed}
        (paths['logs']/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        plot_final_history(pd.DataFrame(rows),paths['logs']/'learning_curves.png',show_plots)
        append_experiment(root/'experiments.csv',{'exp_id':exp_id,'model':'resnet50_agmfa','image_size':256,
            'augmentation':'source: HardShift paired D4+six profiles; target: D4 only, CE weight2' if train_val else 'HardShift paired D4+six profiles',
            'pretrained':True,'val_accuracy':summary['val_accuracy'] if not train_val else '',
            'train_time_min':elapsed,'seed':42,'submission_id':'','public_score':'','notes':cfg['notes']})
        metadata['status']='complete'
    except BaseException as error:
        metadata.update(status='failed',error=f'{type(error).__name__}: {error}')
        emit(metadata['error'])
        raise
    finally:
        (paths['logs']/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return {'model':model,'history':pd.DataFrame(rows),'paths':paths,'summary':summary}


def plot_final_history(history, save_path=None, show=True):
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(17,9))
    groups = [('Source accuracy',['clean_accuracy','hard_accuracy']),('Source CE',['clean_ce','hard_ce']),
              ('JS',['js_loss']),('Total / target CE',['train_loss','target_ce']),('Report / target',['val_loss','val_accuracy','target_accuracy'])]
    for axis,(title,columns) in zip(axes.flat,groups):
        for name in columns:
            if name in history and history[name].notna().any():
                axis.plot(history.epoch,history[name],label=name)
        axis.set_title(title)
        if axis.lines:
            axis.legend()
    rates = [json.loads(row) for row in history.learning_rates]
    for group in rates[0]:
        axes.flat[-1].plot(history.epoch,[row[group] for row in rates],label=group)
    axes.flat[-1].legend()
    axes.flat[-1].set_title('Learning rate used')
    for axis in axes.flat:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path,dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def make_final_submissions(train_config, train_val_config, *, project_root=None, raw_dir=None, device='auto', batch_size=32, num_workers=0):
    """Explicit test operation: exactly two original-view predictions, no fitting."""
    # Test loader is imported only on entering this explicitly called operation.
    from src.data import load_test
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    configs = [load_config(config,root) for config in (train_config,train_val_config)]
    expected_splits = [['train'],['train','val']]
    names = ['submission_agmfa_hardshift_train.csv','submission_agmfa_hardshift_train_val.csv']
    device = get_device(device)
    raws,checkpoints = [],[]
    for cfg,splits in zip(configs,expected_splits):
        validate_final_config(cfg)
        if cfg['final_hardshift']['training_splits']!=splits:
            raise ValueError('Provide train-only then train+val configs')
        raws.append(Path(raw_dir).resolve() if raw_dir else _raw(root,cfg['data']).resolve())
        checkpoint = root/'artifacts/models'/cfg['exp_id']/('last.pt' if splits==['train'] else 'final.pt')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Train the requested final model first: {checkpoint}')
        checkpoints.append(checkpoint)
    if raws[0]!=raws[1] or configs[0]['data']['normalization']!=configs[1]['data']['normalization']:
        raise ValueError('Both models must use identical original test preprocessing/data')
    raw = raws[0]
    sample = pd.read_csv(raw/'sample_submission_example.csv',dtype={'id':str})
    if list(sample.columns)!=['id','label'] or len(sample)!=8299 or sample.id.isna().any() or sample.id.duplicated().any():
        raise ValueError('Expected 8299 unique sample IDs')
    classes = read_classes(raw)
    # Check all output paths before expensive inference, preserve completed runs.
    outputs = []
    for cfg,name in zip(configs,names):
        paths = {'submission':root/'submissions'/name,
                 'probabilities':root/'artifacts/predictions'/cfg['exp_id']/'test_probabilities.npz',
                 'csv':root/'artifacts/predictions'/cfg['exp_id']/'test.csv',
                 'log':root/'artifacts/logs'/cfg['exp_id']/'test_inference.json'}
        if any(path.exists() for path in paths.values()):
            raise FileExistsError('Final inference outputs already exist; preserve them')
        validate_journal(root/'experiments.csv',cfg['exp_id']+'_test')
        outputs.append(paths)
    for cfg,checkpoint,splits,paths in zip(configs,checkpoints,expected_splits,outputs):
        model,saved = load_checkpoint(checkpoint,device,inference_only=True)
        if saved['epoch']!=22 or saved.get('checkpoint_selection')!='fixed_last_epoch' or saved.get('training_splits')!=splits:
            raise ValueError('Inference requires the fixed epoch22 checkpoint of the correct split')
        if saved['config']['model']!=cfg['model'] or saved['config']['seed']!=cfg['seed'] or saved['config']['exp_id']!=cfg['exp_id']:
            raise ValueError('Checkpoint model/seed/run mismatch')
        # CPU inference after Kaggle CUDA training is valid; hardware/logging
        # choices do not alter the learned recipe or the selected checkpoint.
        saved_training = {k:v for k,v in saved['config']['training'].items() if k not in {'device','log_every'}}
        requested_training = {k:v for k,v in cfg['training'].items() if k not in {'device','log_every'}}
        if saved_training!=requested_training:
            raise ValueError('Checkpoint training recipe mismatch')
        if saved['class_labels']!=classes.label.tolist() or saved['config']['data']['normalization']!=cfg['data']['normalization']:
            raise ValueError('Checkpoint class order/normalization mismatch')
        data = dict(cfg['data'],batch_size=batch_size,num_workers=num_workers,pin_memory=device.type=='cuda')
        loader = load_test(raw,data,cfg['seed'])
        prediction = probability_predictions(model,loader,device,classes.label.tolist(),tta='none')
        if len(prediction.ids)!=8299 or set(prediction.ids)!=set(sample.id):
            raise ValueError('Test/sample IDs differ or contain omissions')
        # Save probabilities in SAMPLE order too, so row-wise auditing is direct.
        order = pd.Index(prediction.ids).get_indexer(sample.id)
        prediction.ids=prediction.ids[order]
        prediction.probabilities=prediction.probabilities[order]
        prediction.validate()
        for path in paths.values():
            path.parent.mkdir(parents=True,exist_ok=True)
        save_probabilities(prediction,paths['probabilities'])
        prediction.table().to_csv(paths['csv'],index=False)
        save_submission(prediction.table(),raw/'sample_submission_example.csv',paths['submission'])
        exported = pd.read_csv(paths['submission'],dtype={'id':str})
        np.testing.assert_array_equal(exported.id.to_numpy(),sample.id.to_numpy())
        np.testing.assert_array_equal(exported.label.to_numpy(),prediction.class_labels[prediction.probabilities.argmax(1)])
        report = {'checkpoint':str(checkpoint),'epoch':22,'rows':8299,'unique_ids':8299,'tta':'none',
                  'order':'sample submission','argmax_verified':True,'submission':str(paths['submission'])}
        paths['log'].write_text(json.dumps(report,indent=2),encoding='utf-8')
        append_experiment(root/'experiments.csv',{'exp_id':cfg['exp_id']+'_test','model':'resnet50_agmfa','image_size':256,
            'augmentation':'original only, no TTA','pretrained':True,'val_accuracy':'','train_time_min':'','seed':42,
            'submission_id':paths['submission'].stem,'public_score':'','notes':'Inference only, fixed epoch22; '+str(checkpoint)})
        del model
    return pd.DataFrame([{'submission':str(paths['submission']),'probabilities':str(paths['probabilities'])} for paths in outputs])
