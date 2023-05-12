import pickle
import os
from PIL import Image
from .dataset import Dataset


class TxtDataset(Dataset):
    # pickle the labels. 
    # pickle the annotations in a consistent way.
    # self.img_ann = {'image_name': {width: , height: , bbox: [[xmin, ymin, xmax, ymax], ], label: [...]}}
    # self.label_idx = {'cat': 1, 'dog': 2,...}
    def __init__(self, mode, img_dir, ann_dir=None, ann_cache='txt_cache.pkl', label_cache='txt_label.pkl', 
                transform=None, training_labels: list = [], print_every=100, 
                img_format='.png', 
                 *args, **kwargs):
        
        self.img_dir = img_dir
        self.transform = transform
        if not os.path.exists(ann_cache) or not os.path.exists(label_cache):
            ann_name = os.listdir(ann_dir)
            self.img_ann = {}
            self.label_idx = {}
            index = 1   # 0 - background
            if len(training_labels) == 0:
                print("Training on all found labels...")
            print(f'Processing annotation ... ')
            for idx, ann in enumerate(ann_name):
                ann_file = os.path.join(ann_dir, ann)
                img_name = ann[:-4] + img_format    # '****.txt'[:-4] + '.png'
                img_file = os.path.join(img_dir, img_name)
                img_anns = {}
                image = Image.open(img_file).convert("RGB")
                width, height = image.size
                img_anns['width'] = width
                img_anns['height'] = height
                bboxes = []
                labels = []
                with open(ann_file, 'r') as f:
                    anns = f.readlines()
                for line in anns:
                    info = line.split()
                    if len(training_labels) == 0:
                        labels.append(info[0])
                    elif info[0] in training_labels:
                        labels.append(info[0])
                    else:
                        # label is not in training labels, skip.
                        continue

                    if info[0] not in self.label_idx:
                        self.label_idx[info[0]] = index
                        index += 1

                    xc, yc = float(info[1])*width, float(info[2])*height
                    w, h = float(info[3])*width, float(info[4])*height
                    xmin, ymin = xc - w/2, yc-h/2
                    xmax, ymax = xc + w/2, yc + h/2

                    box = [xmin, ymin, xmax, ymax]
                    if not self.isvoid_box(box):
                        labels = labels[:-1]
                        continue
                    box = self.clip(box, width, height)

                    bboxes.append(box)
                img_anns['bboxes'] = bboxes
                img_anns['labels'] = labels
                img_anns['image_id'] = idx
                self.img_ann[img_name] = img_anns
                if idx % print_every == 0:
                    self.progress_bar(i=idx, total_i=len(ann_name), print_every=print_every)

            self.img_name = list(self.img_ann.keys())

            print('')

        super(TxtDataset, self).__init__(mode=mode, img_dir=img_dir, ann_cache=ann_cache, label_cache=label_cache,
                                         ann_dir=ann_dir, transform=transform)

