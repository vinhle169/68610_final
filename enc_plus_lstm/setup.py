import os
import sys
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms

# initialize COCO API for instance annotations
dataDir = 'opt/cocoapi'
dataType = 'val2014'
instances_annFile = dataDir +'/annotations/instances_{}.json'.format(dataType)
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = dataDir + '/annotations/captions_{}.json'.format(dataType)
coco_caps = COCO(captions_annFile)
print('setup complete!')

