"""Read-only, label-free test diagnostics from four existing submission caches."""
from pathlib import Path
from itertools import combinations
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.inference import file_hash
from src.training.predict import load_probabilities, ProbabilityPredictions

RECIPES=('resnet101_train','resnet101_train_val','ensemble_train','ensemble_train_val')
SHORT={'resnet101_train':'101 T','resnet101_train_val':'101 TV',
       'ensemble_train':'Ens T','ensemble_train_val':'Ens TV',
       'train50':'50 T D4','refit50':'50 TV D4'}


def load_test_analysis(root, suite='submissions_001', raw_dir=None, *, include_sources=False):
    """Align every ID and class column; verify caches against source metadata."""
    root=Path(root).resolve(); raw=Path(raw_dir) if raw_dir else root/'data/raw'
    raw=raw if raw.is_absolute() else root/raw
    if Path(suite).name!=suite or suite in {'','.','..'}: raise ValueError('Invalid suite name')
    classes=pd.read_csv(raw/'classes.csv')
    ids=pd.read_csv(raw/'test.csv',dtype={'id':str}).id.to_numpy(dtype=str)
    if len(set(ids))!=len(ids) or classes.label.duplicated().any(): raise ValueError('Duplicate IDs/classes')
    meta=json.loads((root/'artifacts/logs'/suite/'metadata.json').read_text(encoding='utf-8'))
    if meta['status']!='complete': raise ValueError('Completed suite required')
    for name,key in [('test.csv','test_csv_sha256'),('classes.csv','classes_csv_sha256')]:
        if file_hash(raw/name)!=meta['protocol'][key]: raise ValueError('Suite/data mismatch')
    labels=classes.label.to_numpy(); members={}
    # Four base caches have recorded checksums; verify and reconstruct recipes.
    base={}
    for key in meta['protocol']['sources']:
        info=json.loads((root/'artifacts/logs'/suite/f'{key}.json').read_text(encoding='utf-8'))
        path=root/'artifacts/predictions'/suite/f'{key}.npz'
        if file_hash(path)!=info['probabilities_sha256']: raise ValueError('Member cache checksum mismatch')
        p=load_probabilities(path)
        if set(p.ids)!=set(ids) or set(p.class_labels)!=set(labels) or p.true_labels is not None:
            raise ValueError('Expected unlabeled test cache')
        base[key]=p.probabilities[pd.Index(p.ids).get_indexer(ids)][:,pd.Index(p.class_labels).get_indexer(labels)]
    for recipe in RECIPES:
        p=load_probabilities(root/'artifacts/predictions'/suite/f'{recipe}.npz')
        if set(p.ids)!=set(ids) or set(p.class_labels)!=set(labels) or p.true_labels is not None:
            raise ValueError('Expected unlabeled recipe cache')
        values=p.probabilities[pd.Index(p.ids).get_indexer(ids)][:,pd.Index(p.class_labels).get_indexer(labels)]
        expected=np.stack([base[key].astype(np.float64) for key in meta['protocol']['recipes'][recipe]]).mean(0).astype(np.float32)
        if not np.array_equal(values,expected): raise ValueError('Recipe differs from source caches')
        members[recipe]=ProbabilityPredictions(ids.copy(),labels.copy(),values).validate()
    if include_sources:
        sources={key:ProbabilityPredictions(ids.copy(),labels.copy(),values).validate() for key,values in base.items()}
        return members,classes,sources
    return members,classes


def test_case_table(members,classes):
    """No targets, guessed correctness, calibration fitting or pseudo-labels."""
    if tuple(members)!=RECIPES: raise ValueError('Four ordered recipes required')
    first=members[RECIPES[0]]; names=classes.set_index('label').class_name.to_dict()
    table=pd.DataFrame({'id':first.ids})
    vectors=[]; predictions=[]
    for name,p in members.items():
        p.validate()
        if p.true_labels is not None or not np.array_equal(p.ids,first.ids) or not np.array_equal(p.class_labels,first.class_labels):
            raise ValueError('Unlabeled aligned predictions required')
        order=np.argsort(-p.probabilities,axis=1,kind='stable'); rows=np.arange(len(p.ids))
        for rank in (0,1):
            table[f'{name}_top{rank+1}']=p.class_labels[order[:,rank]]
            table[f'{name}_class{rank+1}']=[names[label] for label in p.class_labels[order[:,rank]]]
            table[f'{name}_p{rank+1}']=p.probabilities[rows,order[:,rank]]
        table[f'{name}_margin']=table[f'{name}_p1']-table[f'{name}_p2']
        entropy=-(p.probabilities*np.log(np.clip(p.probabilities,1e-30,1))).sum(1)
        table[f'{name}_entropy']=entropy
        vectors.append(p.probabilities.astype(np.float64)); predictions.append(table[f'{name}_top1'].to_numpy())
    votes=np.stack(predictions,axis=1)
    table['distinct_answers']=[len(set(row)) for row in votes]
    table['disagree']=table.distinct_answers>1
    table['refit101_changed']=votes[:,0]!=votes[:,1]
    table['refit_ensemble_changed']=votes[:,2]!=votes[:,3]
    table['ensemble_train_changed']=votes[:,0]!=votes[:,2]
    table['ensemble_refit_changed']=votes[:,1]!=votes[:,3]
    stacked=np.stack(vectors); mean=stacked.mean(0)
    entropy_mean=-(mean*np.log(np.clip(mean,1e-30,1))).sum(1)
    mean_entropy=-(stacked*np.log(np.clip(stacked,1e-30,1))).sum(2).mean(0)
    table['js_divergence']=np.maximum(0,entropy_mean-mean_entropy)
    table['min_confidence']=table[[f'{r}_p1' for r in RECIPES]].min(axis=1)
    table['min_margin']=table[[f'{r}_margin' for r in RECIPES]].min(axis=1)
    return table


def select_cases(table,selection='uncertain',*,model='ensemble_train_val',limit=40,seed=42,
                 confidence=.95,pair=None,compare=None):
    if model not in RECIPES or limit<1: raise ValueError('Invalid model/limit')
    if selection=='random': return table.sample(n=min(limit,len(table)),random_state=seed).reset_index(drop=True)
    if selection=='uncertain': result=table.sort_values(f'{model}_margin',kind='stable')
    elif selection=='entropy': result=table.sort_values(f'{model}_entropy',ascending=False,kind='stable')
    elif selection=='disagreement': result=table.loc[table.disagree].sort_values('js_divergence',ascending=False,kind='stable')
    elif selection=='refit101': result=table.loc[table.refit101_changed].sort_values('js_divergence',ascending=False,kind='stable')
    elif selection=='refit_ensemble': result=table.loc[table.refit_ensemble_changed].sort_values('js_divergence',ascending=False,kind='stable')
    elif selection=='confident_disagreement':
        if compare is None:
            mask=table.disagree & table.min_confidence.ge(confidence)
        else:
            if len(compare)!=2 or any(name not in RECIPES for name in compare): raise ValueError('Two valid recipes required')
            left,right=compare
            mask=(table[f'{left}_top1']!=table[f'{right}_top1']) & table[f'{left}_p1'].ge(confidence) & table[f'{right}_p1'].ge(confidence)
        result=table.loc[mask].sort_values('js_divergence',ascending=False,kind='stable')
    elif selection=='confident_agreement':
        result=table.loc[~table.disagree & table.min_confidence.ge(confidence)].sample(frac=1,random_state=seed)
    elif selection=='pair':
        if pair is None or len(pair)!=2: raise ValueError('Two class names required')
        result=table.loc[table[f'{model}_class1'].isin(pair)&table[f'{model}_class2'].isin(pair)].sort_values(f'{model}_margin',kind='stable')
    else: raise ValueError('Unknown selection')
    return result.head(limit).reset_index(drop=True)


def analysis_tables(table,classes):
    summaries=[]; counts=[]; changes=[]; pairwise=[]
    names=classes.set_index('label').class_name.to_dict()
    for recipe in RECIPES:
        summaries.append({'recipe':recipe,'images':len(table),'mean_confidence':table[f'{recipe}_p1'].mean(),
                          'median_margin':table[f'{recipe}_margin'].median(),'margin_below_0.1':int(table[f'{recipe}_margin'].lt(.1).sum()),
                          'confidence_below_0.8':int(table[f'{recipe}_p1'].lt(.8).sum()),'mean_entropy':table[f'{recipe}_entropy'].mean()})
        for label in classes.label:
            rows=table.loc[table[f'{recipe}_top1']==label]
            counts.append({'recipe':recipe,'predicted_class':names[label],'count':len(rows),
                           'mean_confidence':rows[f'{recipe}_p1'].mean() if len(rows) else np.nan,
                           'low_margin_count':int(rows[f'{recipe}_margin'].lt(.1).sum())})
    for left,right in combinations(RECIPES,2):
        changed=table.loc[table[f'{left}_top1']!=table[f'{right}_top1']]
        pairwise.append({'left':left,'right':right,'different_answers':len(changed),'percent':100*len(changed)/len(table)})
        groups=changed.groupby([f'{left}_class1',f'{right}_class1']).size().sort_values(ascending=False)
        for (before,after),count in groups.items():
            changes.append({'left':left,'right':right,'before':before,'after':after,'count':int(count)})
    return {'confidence':pd.DataFrame(summaries),'class_counts':pd.DataFrame(counts),
            'pairwise':pd.DataFrame(pairwise),'transitions':pd.DataFrame(changes)}


def compact_gallery(cases,image_dir,*,title='',columns=5,page_size=20,thumbnail=144):
    """Five small original thumbnails per row, four predictions underneath."""
    if columns<1 or page_size<1 or thumbnail<32: raise ValueError('Invalid gallery dimensions')
    directory=Path(image_dir).resolve(); figures=[]
    if cases.empty: print(title,': no cases'); return figures
    for start in range(0,len(cases),page_size):
        page=cases.iloc[start:start+page_size]; nrows=int(np.ceil(len(page)/columns))
        fig=plt.figure(figsize=(columns*3,nrows*3)); grid=fig.add_gridspec(nrows*2,columns,height_ratios=[1,.52]*nrows,hspace=.22,wspace=.16)
        for index,(_,row) in enumerate(page.iterrows()):
            r,c=divmod(index,columns); ax=fig.add_subplot(grid[2*r,c]); text=fig.add_subplot(grid[2*r+1,c])
            path=(directory/row.id).resolve()
            if not path.is_relative_to(directory): raise ValueError('Image ID escapes test directory')
            with Image.open(path) as source:
                image=source.convert('RGB'); image.thumbnail((thumbnail,thumbnail)); ax.imshow(image,interpolation='nearest')
            ax.axis('off'); ax.set_title(f'{row.id} | answers:{row.distinct_answers}',fontsize=8)
            lines=[f'{SHORT[name]:7s} {row[f"{name}_class1"]} {row[f"{name}_p1"]:.1%}' for name in RECIPES]
            text.axis('off'); text.text(0,1,'\n'.join(lines),va='top',fontsize=7.5,
                                      color='#a63603' if row.disagree else '#333333',fontfamily='monospace')
        fig.suptitle(f'{title}: {start+1}–{start+len(page)} / {len(cases)}',fontsize=11)
        fig.subplots_adjust(left=.02,right=.99,bottom=.025,top=.95)
        plt.show(); figures.append(fig)
    return figures


def probability_detail(image_ids,members,classes,image_dir,*,top_k=5):
    """One original image and four top-k probability panels per requested ID."""
    directory=Path(image_dir).resolve(); names=classes.set_index('label').class_name.to_dict(); figures=[]
    for image_id in image_ids:
        path=(directory/image_id).resolve()
        if not path.is_relative_to(directory): raise ValueError('ID escapes test')
        columns=min(3,len(members)); rows=int(np.ceil(len(members)/columns))
        fig=plt.figure(figsize=(4*(columns+1),3.5*rows)); grid=fig.add_gridspec(rows,columns+1)
        image_ax=fig.add_subplot(grid[:,0])
        with Image.open(path) as source: image_ax.imshow(source.convert('RGB'))
        image_ax.set_title(image_id); image_ax.axis('off')
        axes=[fig.add_subplot(grid[r,c]) for r in range(rows) for c in range(1,columns+1)]
        for ax,(name,p) in zip(axes,members.items()):
            i=pd.Index(p.ids).get_indexer([image_id])[0]
            if i<0: raise ValueError('Unknown image ID')
            order=np.argsort(-p.probabilities[i],kind='stable')[:top_k]; values=p.probabilities[i,order]
            ax.barh([names[label] for label in p.class_labels[order]],values,color='#648caf'); ax.invert_yaxis()
            ax.set_xlim(0,1.15); ax.set_title(SHORT.get(name,name)); ax.tick_params(axis='y',labelsize=8)
            for j,value in enumerate(values): ax.text(value+.015,j,f'{value:.2%}' if value>=.0001 else f'{value:.2e}',va='center',fontsize=8)
        for ax in axes[len(members):]: ax.axis('off')
        fig.tight_layout(); plt.show(); figures.append(fig)
    return figures
