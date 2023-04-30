from .yolox import YOLOX, YOLOXHead
from .yolov3 import YOLOV3PHead, YOLOV3P
from .dense_heads import *
from .point import Point_YOLOV3, Point_YOLOX
from .necks import *
from .backbones import *
from .detectors import *
# from mmdet.models.backbones import Darknet, CSPDarknet
from .builder import build_detector
