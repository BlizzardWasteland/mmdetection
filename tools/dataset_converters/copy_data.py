import os

import cv2
import mmcv
import shutil

import os.path as path

root_path = '/data1/linzhiwei/project/mmdetection/data/VOCdevkit/VOC2012'
txt_path = path.join(root_path, 'ImageSets/Main/')
jpg_path = path.join(root_path, 'JPEGImages/')
splits = ['trainval']
# splits = ['train', 'val', 'test']

for split in splits:
    print(split, 'begin')
    src_path = txt_path + split + '.txt'
    save_path = path.join('/data1/linzhiwei/project/mmdetection/data/my_voc/voc12', split)
    mmcv.mkdir_or_exist(save_path)
    with open(src_path, 'r') as f:
        for ele in f.readlines():
            cur_jpgname = ele.strip()
            total_jpgname = jpg_path + cur_jpgname + '.jpg'
            shutil.copy(total_jpgname, path.join(save_path, cur_jpgname+'.jpg'))
        print(split, 'done')