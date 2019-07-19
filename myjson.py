from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import pickle
import numpy as np
import os
import scipy.sparse
import cv2

# Must happen before importing COCO API (which imports matplotlib)
"""Set matplotlib up."""
import matplotlib
# Use a non-interactive backend
matplotlib.use('Agg')
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from upsnet.config.config import config

from lib.utils.timer import Timer
import upsnet.bbox.bbox_transform as box_utils
import upsnet.mask.mask_transform as segm_utils

from lib.utils.logging import logger

class myJson(object):
    """A class representing a COCO json dataset."""

    def __init__(self, image_dir,anno_file):
        #pass
        if logger:
            # logger.info('Creating: {}'.format(name))
            logger.info('creating......image_oath:{},anno_file:{}'.format(image_dir,anno_file))
        self.image_directory = image_dir
        self.image_prefix = ''
        self.COCO = COCO(anno_file)
        self.debug_timer = Timer()
        #Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]

        self.category_to_id_map = dict(zip(categories, category_ids))   
        self.classes = ['__background__'] + categories       
        self.num_classes = len(self.classes)

        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
             v: k
             for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def read_my_roidb(self,images_path):
        roidb=[]
        images=os.listdir(images_path)
        for image in images:
            image_path=os.path.join(images_path,image)

            image_dict={}
            img=cv2.imread(image_path)
            h,w=img.shape[0],img.shape[1]
            image_name=image

            image_dict['file_name']=image_name   ##name of image
            image_dict['height'] = h
            image_dict['width'] = w
            image_dict['image'] = image_path

            roidb.append(image_dict)
        return roidb
