import os
import xml.etree.ElementTree as ET
import pickle
import torch
from .dataset import Dataset

# Incase your xml format has another label change names on the right side.
# Eg. 'xmin': 'x_min'.
config = {
    'xmin': 'xmin',
    'ymin': 'ymin',
    'xmax': 'xmax',
    'ymax': 'ymax',
    'label': 'name',
    'filename': 'filename',

    'width': 'width',
    'height': 'height',
    'object': 'object',
    'bndbox': 'bndbox'

}


class XmlDataset(Dataset):
    def __init__(self, mode, img_dir, ann_dir=None, ann_cache='xml_cache.pkl', label_cache='xml_label.pkl', 
                transform=None, training_labels: list = [], print_every=200,
                 *args, **kwargs):
        
        self.img_dir = img_dir
        self.transform = transform
        if not os.path.exists(ann_cache) or not os.path.exists(label_cache):
            self.img_ann = {}
            self.label_idx = {}
            index = 1  # 0 for background
            if len(training_labels) == 0:
                print("Training on all found labels...")
            ann_name = os.listdir(ann_dir)
            print('Processing annotations...')
            for idx, ann in enumerate(ann_name):
                xml_file = ET.parse(os.path.join(ann_dir, ann))
                img_anns = {}
                root = xml_file.getroot()
                labels = []
                bboxes = []

                for info in root.iter():
                    if config['filename'] in info.tag:
                        filename = info.text
                    if config['width'] in info.tag:
                        width = int(info.text)
                        img_anns['width'] = width
                    if config['height'] in info.tag:
                        height = int(info.text)
                        img_anns['height'] = height

                    if config['object'] in info.tag:
                        for obj_info in list(info):
                            if config['label'] in obj_info.tag:
                                if len(training_labels) == 0:
                                    # training on all labels
                                    labels.append(obj_info.text)
                                elif obj_info.text in training_labels:
                                    labels.append(obj_info.text)
                                else:
                                    # current label is not found in training_labels, jump
                                    continue
                                if len(training_labels) == 0:
                                    if obj_info.text not in self.label_idx:
                                        self.label_idx[obj_info.text] = index
                                        index += 1
                            if config['bndbox'] in obj_info.tag:
                                bbox = []
                                for bbx_info in list(obj_info):
                                    if config['xmin'] in bbx_info.tag:
                                        bbox.append(float(bbx_info.text))
                                    if config['ymin'] in bbx_info.tag:
                                        bbox.append(float(bbx_info.text))
                                    if config['xmax'] in bbx_info.tag:
                                        bbox.append(float(bbx_info.text))
                                    if config['ymax'] in bbx_info.tag:
                                        bbox.append(float(bbx_info.text))
                                if not self.isvoid_box(bbox):
                                    labels = labels[:-1]
                                    continue 
                                bbox = self.clip(bbox, width, height)

                                bboxes.append(bbox)

                img_anns['bboxes'] = bboxes
                img_anns['labels'] = labels
                img_anns['image_id'] = idx
                assert len(img_anns['bboxes']) == len(img_anns['labels']), \
                    f'For each bboxes there should be a label. ' \
                    f'{filename} have {len(labels)} labels and {len(bboxes)} boxes.'
                self.img_ann[filename] = img_anns

                if idx % print_every == 0:
                    self.progress_bar(i=idx, total_i=len(ann_name), print_every=print_every)
            self.img_name = list(self.img_ann.keys())

            print('')
            
        super(XmlDataset, self).__init__(mode=mode, img_dir=img_dir, ann_cache=ann_cache, label_cache=label_cache,
                                         ann_dir=ann_dir, transform=transform)


