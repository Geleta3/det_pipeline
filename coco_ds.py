
from pycocotools.coco import COCO
import os
import pickle
from .dataset import Dataset
from PIL import Image


class CocoDataset(Dataset):
    def __init__(self, mode, img_dir, ann_dir=None, ann_cache='coco_ann.pkl', label_cache='coco_label.pkl', 
                transform=None, training_labels: list = [],  print_every=200, 
                 *args, **kwargs):

        self.img_dir = img_dir
        self.transform = transform
        if not os.path.exists(ann_cache) or not os.path.exists(label_cache):
            path = os.path.join(ann_dir, 'annotations', f'instances{mode if mode in "train" else "val"}2017.json')
            self.coco = COCO(path)
            img_ids = self.coco.getImgIds()
            if len(training_labels) > 0:
                training_labels = self.coco.getCatIds(training_labels)
            else:
                training_labels = self.coco.getCatIds()

            self.label_idx = dict(zip(training_labels, [i for i in range(1, len(training_labels)+1)]))
            self.img_ann = {}

            print('Processing annotations...')
            for i, img_id in enumerate(img_ids):
                file_name = str(img_id).zfill(12) + '.jpg'
                image = Image.open(os.path.join(self.img_dir, file_name))
                width, height = image.size

                all_ind = self.coco.getAnnIds(imgIds=img_id)
                objects = self.coco.loadAnns(all_ind)
                bboxes = []
                labels = []
                for obj in objects:
                    if obj['category_id'] not in training_labels: continue
                    if not self.isvoid_box(obj['bbox']): continue
                    bbox = self.clip(obj['bbox'], width=width, height=height)
                    labels.append(obj['category_id'])
                    bboxes.append(bbox)
                if len(labels) <= 0: continue
                
                self.img_ann[file_name] = {'width': width, 
                                           'height': height, 
                                           'bboxes': bboxes, 
                                           'labels': labels, 
                                           'image_id': img_id}
                if i%print_every == 0:
                    self.progress_bar(i=i, total_i=len(img_ids), print_every=print_every)

            print()

        super(CocoDataset, self).__init__(mode=mode, img_dir=img_dir, ann_cache=ann_cache, label_cache=label_cache,
                                          ann_dir=ann_dir, transform=transform)





