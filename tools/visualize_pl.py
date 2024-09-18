import logging
import os
import copy
import torch
import torch.nn.functional as F
import torchvision.ops as ops

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
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
    UAVDTGtaDetectionEvaluator,
    GTAV10KDetectionEvaluator,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from pynvml import *
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import random
import cv2
import pdb

logger = logging.getLogger("detectron2")
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

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
    if evaluator_type == "uavdtgta":
        return UAVDTGtaDetectionEvaluator(dataset_name)
    if evaluator_type == "gtav10k":
        return GTAV10KDetectionEvaluator(dataset_name)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

# =====================================================
# ================== Pseduo-labeling ==================
# =====================================================
def threshold_bbox(proposal_bbox_inst, thres=0.7, proposal_type="roih"):
    if proposal_type == "rpn":
        valid_map = proposal_bbox_inst.objectness_logits > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
            valid_map
        ]
    elif proposal_type == "roih":
        valid_map = proposal_bbox_inst.scores > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

    return new_proposal_inst

def process_pseudo_label(proposals_rpn_k, cur_threshold, proposal_type, psedo_label_method=""):
    list_instances = []
    num_proposal_output = 0.0
    for proposal_bbox_inst in proposals_rpn_k:
        # thresholding
        if psedo_label_method == "thresholding":
            proposal_bbox_inst = threshold_bbox(
                proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
            )
        else:
            raise ValueError("Unkown pseudo label boxes methods")
        num_proposal_output += len(proposal_bbox_inst)
        list_instances.append(proposal_bbox_inst)
    num_proposal_output = num_proposal_output / len(proposals_rpn_k)
    return list_instances, num_proposal_output

def draw_pl(cfg, model_student, model_teacher, args):
    with EventStorage(0) as storage:
        data_loader = build_detection_train_loader(cfg)
        evaluator_type = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).evaluator_type
        len_data_loader = len(data_loader.dataset.dataset.dataset)
        model_teacher.eval()
        model_student.train()
        start_iter, max_iter = 0, len_data_loader
        os.makedirs(os.path.join("visualization", evaluator_type, "pseudo_labels"), exist_ok=True)

        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            # Process data on GPU, but move results to CPU as soon as possible
            with torch.no_grad():
                _, teacher_features, teacher_proposals, teacher_results = model_teacher(data, mode="train", iteration=iteration)

            # Process pseudo labels and filter
            teacher_pseudo_results, _ = process_pseudo_label(teacher_results, 0.7, "roih", "thresholding")
            _, t_indices, t_boxes, info = model_student(
                data, cfg, model_teacher, teacher_features,
                teacher_proposals, teacher_pseudo_results, mode="train", iou_threshold=0.4
            )
            
            # Visualize high confidence boxes
            img1 = data[0]["image_weak"].permute(1, 2, 0).cpu().numpy()
            img1 = np.ascontiguousarray(img1, dtype=np.uint8)
            image_id = data[0]["image_id"]
            hpl_box = teacher_pseudo_results[0].gt_boxes.tensor.detach().cpu().numpy()  # Move to CPU
            for i in range(len(hpl_box)):
                cv2.rectangle(img1, (int(hpl_box[i][0]), int(hpl_box[i][1])),
                              (int(hpl_box[i][2]), int(hpl_box[i][3])), (113, 179, 60), 2)
            
            # Visualize before filtering
            if t_indices is not None:
                img2 = data[0]["image_weak"].permute(1, 2, 0).cpu().numpy()
                img2 = np.ascontiguousarray(img2, dtype=np.uint8)
                filtered_box = t_boxes[t_indices].tensor.detach().cpu().numpy()  # Move to CPU
                for i in range(len(filtered_box)):
                    cv2.rectangle(img2, (int(filtered_box[i][0]), int(filtered_box[i][1])),
                                  (int(filtered_box[i][2]), int(filtered_box[i][3])), (0, 140, 255), 2)
            
            # Visualize after filtering
            if "t_indices_filtered" in info.keys():                  
                img3 = data[0]["image_weak"].permute(1, 2, 0).cpu().numpy()
                img3 = np.ascontiguousarray(img3, dtype=np.uint8)
                filtered_box = t_boxes[info["t_indices_filtered"]].tensor.cpu()  # Move to CPU
                for i in range(len(filtered_box)):
                    cv2.rectangle(img3, (int(filtered_box[i][0]), int(filtered_box[i][1])),
                                    (int(filtered_box[i][2]), int(filtered_box[i][3])), (71, 99, 255), 2)                

            if t_indices is not None and "t_indices_filtered" in info.keys():
                img_concat = np.concatenate((img1, img2, img3), axis=1)
                cv2.imwrite(os.path.join("visualization", evaluator_type, "pseudo_labels", f"pl_{image_id}.png"), img_concat)

            # Clear cache to free up GPU memory
            del teacher_features, teacher_proposals, teacher_results, teacher_pseudo_results
            del t_indices, t_boxes, info
            torch.cuda.empty_cache()
        
        return

def setup(args):

    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    
    seed = 1000
    print("SEED: ", seed)
    print("Output_dir: ", cfg.OUTPUT_DIR)
    #seed = 1122
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "0"
    return cfg


def main(args):
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = "student_sfda_RCNN_ablation"
    cfg.SOURCE_FREE.TYPE = True
    cfg.SOURCE_FREE.MODE = True
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.freeze()
    model_student = build_model(cfg)
    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = "teacher_sfda_RCNN"
    cfg.freeze()
    model_teacher = build_model(cfg)
    DetectionCheckpointer(model_student, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    DetectionCheckpointer(model_teacher, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    return draw_pl(cfg, model_student, model_teacher, args)

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