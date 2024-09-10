from detectron2.utils.visualizer import Visualizer
import logging
import os
import copy
import torch.optim as optim
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision.ops as ops
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import pickle
import pandas as pd

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    ClipartDetectionEvaluator,
    WatercolorDetectionEvaluator,
    CityscapeDetectionEvaluator,
    FoggyDetectionEvaluator,
    CityscapeCarDetectionEvaluator,
    BddDetectionEvaluator,
    KaistDetectionEvaluator,
    KaistPersonDetectionEvaluator,
    FLIRDetectionEvaluator,
    VisDroneDetectionEvaluator,
    VisDroneDotaCBTDetectionEvaluator,
    DOTADetectionEvaluator,
    DOTAgtaDetectionEvaluator,
    UAVDTDetectionEvaluator,
    UAVDTDotaDetectionEvaluator,
    GTAV10KDetectionEvaluator,
    
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

import pdb
import cv2
from pynvml import *
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.structures import pairwise_iou
from detectron2.data.detection_utils import convert_image_to_rgb

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import random
import sys
from detectron2.layers import cat

logger = logging.getLogger("detectron2")
from matplotlib import pyplot as plt

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if evaluator_type == "clipart":
        return ClipartDetectionEvaluator(dataset_name)
    if evaluator_type == "watercolor":
        return WatercolorDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape":
        return CityscapeDetectionEvaluator(dataset_name)
    if evaluator_type == "foggy":
        return FoggyDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape_car":
        return CityscapeCarDetectionEvaluator(dataset_name)
    if evaluator_type == "bdd100k":
        return BddDetectionEvaluator(dataset_name)
    if evaluator_type == "kaist_viz":
        return KaistDetectionEvaluator(dataset_name)
    if evaluator_type == "kaist_tr":
        return KaistDetectionEvaluator(dataset_name)
    if evaluator_type == "kaist_viz_person":
        return KaistPersonDetectionEvaluator(dataset_name)
    if evaluator_type == "kaist_tr_person":
        return KaistPersonDetectionEvaluator(dataset_name)
    if evaluator_type == "flir":
        return FLIRDetectionEvaluator(dataset_name)
    if evaluator_type == "visdrone":
        return VisDroneDetectionEvaluator(dataset_name)
    if evaluator_type == "visdronedota":
        return VisDroneDotaCBTDetectionEvaluator(dataset_name)
    if evaluator_type == "dota":
        return DOTADetectionEvaluator(dataset_name)
    if evaluator_type == "dotagta":
        return DOTAgtaDetectionEvaluator(dataset_name)
    if evaluator_type == "uavdt":
        return UAVDTDetectionEvaluator(dataset_name)
    if evaluator_type == "uavdtdota":
        return UAVDTDotaDetectionEvaluator(dataset_name)
    if evaluator_type == "gtav10k":
        return GTAV10KDetectionEvaluator(dataset_name)

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

def do_draw(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        os.makedirs(f"./visualization/{evaluator_type}", exist_ok=True)
        results = inference_on_dataset(model, data_loader, evaluator, draw=True, dirname=evaluator_type)
    
    return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    model.eval()
    return do_draw(cfg, model)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
