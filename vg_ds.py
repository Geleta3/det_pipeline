from .dataset import Dataset
import os 
import json
from PIL import Image
import pickle 

unnecessary_labels_ = ['a', 'the', 'is', 'an', 'two', 'adult', 
                      'white', 'it', 'large', 'down',
                      'top', 'group', 'up', 'front',
                      'black', 'red', 'blue', 'green',
                      'side', 'game', 'yellow', 'brown',
                      'this', 'back', 'view', 'bunch',
                      'lot', 'day', 'piece', 'something',
                      'runway', 'he', 'dark',  'end', 'left', 
                      'color', 'log','objects', 'shape', 
                      'daytime', 'strip', 'city', 'background',
                      'dirt', 'image', 'inside', 'outside', 
                      ]

class VgDataset(Dataset):
    def __init__(self, mode, img_dir, ann_dir=None, ann_cache='txt_cache.pkl', label_cache='txt_label.pkl', 
                transform=None, training_labels: list = [], print_every=100,
                
                file_type = 'attributes', threshold=150, 
                *args, **kwargs):
                
        self.img_dir = img_dir
        self.transform = transform
        self.unnecessary_labels = unnecessary_labels_
        self.file_type = file_type
        if not os.path.exists(ann_cache) or not os.path.exists(label_cache):
            ann_path = os.path.join(ann_dir, f'{file_type}.json')
            with open(ann_path, 'r') as f:
                self.anns = json.load(f)

            training_labels = self.get_labels(training_labels=training_labels, threshold=threshold).keys()
            self.label_idx = dict(zip(training_labels, [i for i in range(0, len(training_labels))]))
            self.img_ann = {}
            att_count = 0
            attno_count = 0
            nobbox_count = 0
            print("Processing Annotations...")
            print("__This may take some minutes__")
            for i, img in enumerate(self.anns):
                img_ids = img['image_id']
                img_name = str(img_ids) + '.jpg'
                img_path = os.path.join(self.img_dir, img_name)
                image = Image.open(img_path)
                w, h = image.size

                objects = img[f'{file_type}']
                bboxes = []
                labels = []
                attributes = []
                for obj in objects:
                    name = obj['names'][0]
                    if name in training_labels:
                        box = self._transform_box(obj['x'], obj['y'], obj['w'], obj['h'])
                        if not self.isvoid_box(box): continue 
                        box = self.clip(box, w, h)
                        bboxes.append(box)
                        labels.append(name)
                        if file_type in obj: #== 'attributes':
                            attributes.append(obj['attributes'])
                            att_count += 1
                        else:
                            attno_count += 1
                            # print('Not found Attributes')
                
                if len(bboxes) > 0:
                    self.img_ann[img_name] = {  'width': w, 
                                                'height': h, 
                                                'bboxes':bboxes, 
                                                'labels': labels, 
                                                "attributes":attributes, 
                                                'image_id': int(img_ids)
                                            }
                else:
                    nobbox_count += 1
                #     print('Image with no training label.')
                if i%print_every==0:
                    self.progress_bar(i, len(self.anns), print_every)
                    # print('\r', f'Processing annotation... {i}/{len(self.anns)}', end='')
            
            self.img_name = list(self.img_ann.keys())
            if file_type == 'attributes':
                print(f"{att_count} attributes found {attno_count} attributes not found / {attno_count + att_count}.")
            elif file_type == 'objects':
                print(f'{nobbox_count} images have no bboxes.')
        
        super().__init__(mode=mode, img_dir=img_dir, transform=transform, 
                        ann_cache=ann_cache, label_cache=label_cache, ann_dir=ann_dir)
    
    def get_labels(self, training_labels=[], threshold=150):
        """_summary_

        Args:
            training_labels (tuple, optional): if training label is provided, it will only train on them
                    Else it will train on all labels which has frequency of {threshold}. Defaults to ().
            threshold (int, optional): . Defaults to 150.

        Returns:
            Dict: label with frequency of these threshold will be returned.
        """

        labels_count = {}
        print('These are unnecessary labels: \n', self.unnecessary_labels)
        print('Processing labels...')
        for id, img in enumerate(self.anns):
            objs = img['objects']
            for obj in objs:
                names = obj['names']
                for name in names[:1]:
                    if name not in labels_count:
                        labels_count[name] = 1
                    else:
                        labels_count[name] += 1
            if id%10000==0:
                print("\r", f"Finished... {id}/{len(self.anns)}", end='')
        print()
        if len(training_labels) > 0:
            labels = [label for label in labels_count if label in training_labels]
            not_found_labels = []
            for label in training_labels:
                if label not in labels_count: not_found_labels.append(label)
            
            assert len(not_found_labels) != len(training_labels), "None of labels you provide are in the dataset"
            
            if len(not_found_labels) > 0:
                print("These labels are not found in the dataset.\n", not_found_labels)
                print("Training on found labels...")
            
            training_labels = [label for label in training_labels if label not in not_found_labels]
            
        else:       
            print('Getting all labels...')
            training_labels = [label for label in labels_count if labels_count[label] >= threshold]
        
        sorted_labels = sorted(training_labels, reverse=True, key= lambda label: labels_count[label])
        labels_count = dict(zip(sorted_labels, [labels_count[label] for label in sorted_labels]))
        # labels = dict(zip(sorted_labels, [i for i in range(1, len(sorted_labels)+1)]))
        
        return labels_count

    def _transform_box(self, *obj):
        xmin, ymin = obj[0], obj[1]
        xmax, ymax = xmin + obj[2], ymin + obj[3]
        return [xmin, ymin, xmax, ymax] 

                


            