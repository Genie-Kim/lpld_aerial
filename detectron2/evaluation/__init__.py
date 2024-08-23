# Copyright (c) Facebook, Inc. and its affiliates.
from .cityscapes_evaluation import CityscapesInstanceEvaluator, CityscapesSemSegEvaluator
from .coco_evaluation import COCOEvaluator
from .rotated_coco_evaluation import RotatedCOCOEvaluator
from .evaluator import DatasetEvaluator, DatasetEvaluators, inference_context, inference_on_dataset, inference_on_corruption_dataset, inference_on_cropped_dataset
from .lvis_evaluation import LVISEvaluator
from .panoptic_evaluation import COCOPanopticEvaluator
from .pascal_voc_evaluation import PascalVOCDetectionEvaluator
from .sem_seg_evaluation import SemSegEvaluator
from .testing import print_csv_format, verify_results

from .clipart_evaluation import ClipartDetectionEvaluator
from .watercolor_evaluation import WatercolorDetectionEvaluator
from .cityscape_evaluation import CityscapeDetectionEvaluator
from .foggy_evaluation import FoggyDetectionEvaluator
from .cityscape_car_evaluation import CityscapeCarDetectionEvaluator
from .sim10k_evaluation import Sim10kDetectionEvaluator
from .kaist_evaluation import KaistDetectionEvaluator
from .kaist_person_evaluation import KaistPersonDetectionEvaluator
from .bdd100k_evaluation import BddDetectionEvaluator
from .flir_evaluation import FLIRDetectionEvaluator
from .dota_evaluation import DOTADetectionEvaluator
from .visdrone_evaluation import VisDroneDetectionEvaluator
from .visdronedota_evaluation import VisDroneDotaCBTDetectionEvaluator
from .uavdt_evaluation import UAVDTDetectionEvaluator
from .uavdtdota_evaluation import UAVDTDotaDetectionEvaluator
from .gtav10k_evaluation import GTAV10KDetectionEvaluator

__all__ = [k for k in globals().keys() if not k.startswith("_")]
