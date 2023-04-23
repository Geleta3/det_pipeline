import torch.utils.data as data


def collate_fn(batch):
    return tuple(zip(*batch))


def dataloader(fileformat, mode, batch_size,  img_dir, ann_dir, transform, 
                num_workers, ann_cache='ann.pkl', label_cache='label.pkl', 
               training_labels: list=[], print_every=100, 
               
               # special for coco

               # special for txt
               img_format='.png', 

               # special for xml 
               
               # special for vg
               file_type='attributes', threshold=150
               ):
    if fileformat == 'xml':
        from .xml_ds import XmlDataset as Dataset
    elif fileformat == 'txt':
        from .txt_ds import TxtDataset as Dataset
    elif fileformat == 'coco':
        from .coco_ds import CocoDataset as Dataset
    elif fileformat == 'vg':
        from .vg_ds import VgDataset as Dataset 
    else:
        raise ValueError("fileformat should be in 'xml', 'txt', 'coco', 'vg")

    dataset = Dataset(mode=mode,
                      ann_dir=ann_dir,
                      img_dir=img_dir,
                      transform=transform,
                      ann_cache=ann_cache,
                      label_cache=label_cache,
                      training_labels=training_labels,
                      print_every=print_every, 
                      
                      # txt
                      img_format=img_format, 

                      # VG
                      file_type=file_type,
                      threshold=threshold
                      )

    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)

    return dataloader


