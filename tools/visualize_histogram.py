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
    VisDroneGtaDetectionEvaluator,
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
    if evaluator_type == "visdronegta":
        return VisDroneGtaDetectionEvaluator(dataset_name)
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

def plot_histogram(cfg, model_student, model_teacher, args):
    with EventStorage(0) as storage:
        data_loader = build_detection_train_loader(cfg)
        len_data_loader = len(data_loader.dataset.dataset.dataset)
        model_teacher.eval()
        model_student.train()
        start_iter, max_iter = 0, len_data_loader
        conf_per_iou = {"confidence": [], "iou": []}
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            with torch.no_grad():
                _, teacher_features, teacher_proposals, teacher_results = model_teacher(data, mode="train", iteration=iteration)

            teacher_pseudo_results, _ = process_pseudo_label(teacher_results, 0.7, "roih", "thresholding")
            gt = data[0]["instances"].gt_boxes.tensor
            gt_cls = data[0]["instances"].gt_classes
            _, t_indices, t_boxes, info = model_student(data, cfg, model_teacher, teacher_features,
                                                        teacher_proposals, teacher_pseudo_results, mode="train", iou_threshold=0.0) # only extract non-overlapping boxes
            iou_matrix = ops.box_iou(teacher_pseudo_results[0].gt_boxes.tensor.cpu(), gt)
            for i in range(iou_matrix.shape[0]):
                if iou_matrix[i].sum() > 0:
                    max_iou = torch.max(iou_matrix[i]).item()
                    max_idx = iou_matrix[i].argmax()
                    og_cls = gt_cls[max_idx].item()
                    pr_cls = teacher_pseudo_results[0].gt_classes[i].item()
                    if pr_cls == og_cls:
                        confidence = teacher_pseudo_results[0].scores[i].item()
                        conf_per_iou["confidence"].append(confidence)
                        conf_per_iou["iou"].append(max_iou)

            if t_indices is not None:
                iou_matrix = ops.box_iou(t_boxes[t_indices].tensor.cpu(), gt)
                for i in range(iou_matrix.shape[0]):
                    if iou_matrix[i].sum() > 0:
                        max_iou = torch.max(iou_matrix[i]).item()
                        max_idx = iou_matrix[i].argmax()
                        # og_cls = gt_cls[iou_matrix[i].argmax()]
                        # pr_cls = info["t_logits"][t_indices][i][:-1].argmax()
                        confidence = F.softmax(info["t_logits"][t_indices][i], dim=0)[:-1].max().item()

                        conf_per_iou["confidence"].append(confidence)
                        conf_per_iou["iou"].append(max_iou)

            ##############################################

            print(f"Iteration: {iteration}/{max_iter}")
            if len(conf_per_iou["confidence"]):
                x_data = np.array(conf_per_iou["confidence"])
                y_data = np.array(conf_per_iou["iou"])
                
                # Define the number of bins along each axis
                bins = [100, 50] # Adjust this based on your data distribution and the resolution you want

                # Create a 2D histogram of the data
                hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins, range=[[x_data.min(), x_data.max()], [y_data.min(), y_data.max()]])

                # Use np.log1p to scale the histogram values for better visualization, if needed

                # Apply Gaussian filter for smoothing
                sigma = [1.5, 1.5]  # Adjust sigma values to control smoothing; [sigma_x, sigma_y]
                hist = gaussian_filter(hist, sigma=sigma)
                hist_log = np.log10(hist+1)

                plt.figure(figsize=(8, 4))

                # Plot the heatmap. Note that we need to transpose hist since the first index corresponds to the y-axis
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                vmin, vmax = hist_log.min(), hist_log.max()
                plt.imshow(hist_log.T, extent=extent, origin='lower', cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
                plt.colorbar()

                plt.xlabel('Confidence')
                plt.ylabel('IoU with GT')
                plt.tight_layout()

                plt.savefig("density_histogram.png")
                plt.close()
                
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
    return plot_histogram(cfg, model_student, model_teacher, args)

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