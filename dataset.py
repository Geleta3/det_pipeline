import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle

import torch
import torch.utils.data as data
from PIL import Image

# Parent class 
class Dataset(data.Dataset):
    """
    >> Children Class constructor implements and will contain the following mainly.

    self.img_dir: is directory/folders of the images.
    self.img_name: contains the file name of the image.
    self.img_ann: is a dictionary containing about the image and objects.
                Format: {'width': int, 'height': int, 'bboxes': [[xmin, ymin, xmax, ymax]], 'labels': [], 'image_id': int}
    self.transform: transform for the image.

    Don't forget to call the the constructor (__init__) at the end of your constructor.
    """
    def __init__(self, mode, img_dir, transform=None, ann_cache='ann.pkl', label_cache='label.pkl', ann_dir=''):

        self.img_dir = img_dir
        self.transform = transform
        self.is_test_mode(mode, label_cache)  # raise assertion if label cache is not provided during test or valid.

        if not os.path.exists(ann_cache):
            assert os.path.exists(ann_dir), "Provide annotation dir."

        if os.path.exists(ann_cache) and os.path.exists(label_cache):
            with open(ann_cache, 'rb') as f:
                self.img_ann = pickle.load(f)
                self.img_name = list(self.img_ann.keys())
            with open(label_cache, 'rb') as f:
                self.label_idx = pickle.load(f)
        else:
            with open(ann_cache, 'wb') as f:
                pickle.dump(self.img_ann, f)
            with open(label_cache, 'wb') as f:
                pickle.dump(self.label_idx, f)
            self.img_name = list(self.img_ann.keys())
            print('\n', f'annotations successfully cached @ {ann_cache} and label @ {label_cache}')

    def __getitem__(self, item):
        img_name = self.img_name[item]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        bboxes = self.img_ann[img_name]['bboxes']
        labels = self.img_ann[img_name]['labels']
        image_id = self.img_ann[img_name]['image_id']
        labels = [self.label_idx[label] for label in labels]

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.as_tensor(image_id)
        is_crowd = torch.tensor([0]*len(bboxes))
        area = (bboxes[..., 3] - bboxes[..., 1]) * (bboxes[..., 2] - bboxes[..., 0])

        target = {'boxes': bboxes, 
                  'labels': labels, 
                  'area': area, 
                  'image_id': image_id, 
                  'iscrowd': is_crowd,
                  'file_name': img_name
                  }
        
        # If mask, keypoints and attributes are provided, 
        # they should be processed in transform. 
        if 'masks' in self.img_ann[img_name]:
            target['masks'] = self.img_ann[img_name]['masks']
        if 'keypoints' in self.img_ann[img_name]:
            target['keypoints'] = self.img_ann[img_name]['masks']
            
        if 'attributes' in self.img_ann[img_name].keys():
            target['attributes'] = self.img_ann[img_name]['attributes']

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.img_name)

    def is_test_mode(self, mode, label_cache):
        if mode in ['test', 'valid']:
            assert os.path.exists(label_cache), "Labels with their indexes should be first generated before test."

    @staticmethod
    def isvoid_box(box):
        pos = True
        for pnt in box:
            pos = pos and (pnt >= 0)

        return pos and (box[0] < box[2]) and (box[1] < box[3])

    @staticmethod
    def clip(box, width, height):
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(width, box[2])
        box[3] = min(height, box[3])

        return box

    @staticmethod
    def progress_bar(i, total_i, print_every=1000):
        # n = min (target) + (xi - min(x))/(delta)*(target_max - target_min)
        if i%print_every == 0:
            t_min, t_max = 0, 51
            x_min, x_max = 0, total_i

            percent = int(t_min + (i-x_min)/(x_max-x_min)*(t_max - t_min))
            rem = t_max - percent
            print('\r', '[', '#'*percent, ' '*rem, ']', percent*2, '%', end='')





