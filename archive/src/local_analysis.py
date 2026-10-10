"""Visualize whole-frame and crop probabilities from saved validation diagnostics."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from archive.src.local_inference import CROPS, crop_boxes
from archive.src.training.predict import load_probabilities


def show_local_cases(image_ids, raw_dir, predictions_dir, classes, *, crop_size=160, page_size=3):
    names=classes.set_index('label').class_name.to_dict()
    predictions={name:load_probabilities(Path(predictions_dir)/f'{name}.npz') for name in ('none',)+CROPS}
    positions={name:{image_id:i for i,image_id in enumerate(p.ids)} for name,p in predictions.items()}
    ids=list(image_ids)
    if page_size<1:raise ValueError("Positive page_size required")
    figures=[]
    def probability_text(value):
        if 0<value<.0001:return f'{100*value:.2e}%'
        return f'{value:.6%}' if value>=.9999 else f'{value:.2%}'
    for start in range(0,len(ids),page_size):
        page=ids[start:start+page_size]
        fig,axes=plt.subplots(len(page),6,figsize=(21,6*len(page)),squeeze=False)
        for row_axes,image_id in zip(axes,page):
            directory=(Path(raw_dir)/'val').resolve();path=(directory/image_id).resolve()
            if not path.is_relative_to(directory):raise ValueError("Image ID escapes val")
            with Image.open(path) as source:
                image=source.convert('RGB')
                images=[image]+[image.crop(box) for box in crop_boxes(image.size,crop_size)]
            for ax,name,view in zip(row_axes,('none',)+CROPS,images):
                p=predictions[name];i=positions[name][image_id];truth=p.true_labels[i]
                order=np.argsort(-p.probabilities[i],kind='stable')
                top=[f'{names[p.class_labels[j]]}: {probability_text(p.probabilities[i,j])}' for j in order[:3]]
                true_index=int(np.flatnonzero(p.class_labels==truth)[0])
                rank=int(np.flatnonzero(order==true_index)[0])+1
                ax.imshow(view,interpolation='nearest');ax.axis('off')
                ax.set_title(f'{image_id} | {name}\nTrue: {names[truth]}\n'+ '\n'.join(top)+f'\nTrue rank: {rank}; p: {probability_text(p.probabilities[i,true_index])}',
                             fontsize=9,color='#238b45' if p.class_labels[order[0]]==truth else '#a63603')
        fig.suptitle(f'Full frame and crops: {start+1}–{start+len(page)} / {len(ids)}',fontsize=12)
        fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.91,hspace=.9,wspace=.15)
        plt.show();figures.append(fig)
    return figures
