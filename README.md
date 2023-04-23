# Dataset Pipeline for Object Detection Based on Pytorch
Data pipelining for object detection. 

This repo aims at unifying different dataset pipelining for object detection
based on pytorch framework. Currently this repo contains three formats, for MSCOCO, 
annoation in txt (Yolo1 format) and xml format. Pipeline for Visual Genome dataset is under way. 

All datasets inherit the class Dataset in dataset.py file and **overwrite** the **__init__** method. 
If you have different format of dataset you can inherit the class <Dataset> and overwrite the __init__. 
In this method these parameters should be defined: 
  * self.img_dir
  * self.transform if there is
  * self.img_ann
  * self.label_idx - Dictionary that maps the label and number. 
  * self.img_name - The name of the images. You can use <os.listdir>

Further documentation will be added. 
If you have any bugs or question feel free to pull request. 
