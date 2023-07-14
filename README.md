# Dataset Pipeline for Object Detection Based on Pytorch
Data pipelining for object detection. 

This repo aims at unifying different dataset pipelining for object detection
based on pytorch framework. Currently this repo contains three formats, for MSCOCO, 
annoation in txt (Yolo1 format) Visual Genome and xml format.

All datasets inherit the class Dataset in ```dataset.py``` file and **overwrite** the ```__init__``` method. 
If you have different format of dataset you can inherit the class <Dataset> and overwrite the __init__. 
In this method these parameters should be defined: 
  * self.img_dir
  * self.transform if there is
  * self.img_ann - Dict(width, height, bboxes, labels, image_id)
  * self.label_idx - Dictionary that maps the label and number. 

After you define these parameters call '__init__' of super class at the end. 

One of the benefits in these repo is that it enable the annotations to be cached so that the next time you start training 
no time is wasted in uploading the datasets. 

Further documentation will be added. 
If you have any bugs or question feel free to create issue
