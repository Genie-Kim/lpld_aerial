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
    UAVDTGtaDetectionEvaluator,
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
    elif proposal_type == "rpn_scores":
        valid_map = proposal_bbox_inst.scores > thres
         # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.scores = proposal_bbox_inst.scores[
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

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.996):
    if comm.get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model_student.state_dict().items()
        }
    else:
        student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    return new_teacher_dict

def test_sfda(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        cfg.defrost()
        cfg.SOURCE_FREE.TYPE = False
        cfg.freeze()
        test_data_loader = build_detection_test_loader(cfg, dataset_name) #list sized batch_size with dict 'file_name', 'image_id', 'height', 'width', 'image'
        test_metadata = MetadataCatalog.get(dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, test_data_loader, evaluator)
        results[dataset_name] = results_i

        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
            cls_names = test_metadata.get("thing_classes")
            cls_aps = results_i['bbox']['class-AP50']
            for i in range(len(cls_aps)):
                logger.info("AP for {}: {}".format(cls_names[i], cls_aps[i]))

    if len(results) == 1:
        results = list(results.values())[0]
    return results


def validation_sfda(cfg, model_student, model_teacher, storage, writers, iteration, epoch):
    model_student.eval()
    
    print("Student model testing@", epoch)
    results = test_sfda(cfg, model_student)
    class_ap50 = results['bbox'].pop('class-AP50')
    class_ap50 = {f'stu_class-AP50/{k}': v for k,v in enumerate(class_ap50)}
    student_results = {f'student/{k}': v for k,v in results['bbox'].items()}
    storage.put_scalars(**student_results)
    storage.put_scalars(**class_ap50)
    
    print("Teacher model testing@", epoch)            
    results = test_sfda(cfg, model_teacher)
    class_ap50 = results['bbox'].pop('class-AP50')
    class_ap50 = {f'tea_class-AP50/{k}': v for k,v in enumerate(class_ap50)}
    teacher_results = {f'teacher/{k}': v for k,v in results['bbox'].items()}
    storage.put_scalars(**teacher_results)
    storage.put_scalars(**class_ap50)

    for writer in writers:
        writer.write()
        
    torch.save(model_teacher.state_dict(), cfg.OUTPUT_DIR + "/model_teacher_{}_{}.pth".format(iteration, epoch))
    torch.save(model_student.state_dict(), cfg.OUTPUT_DIR + "/model_student_{}_{}.pth".format(iteration, epoch))
    model_student.train()
                

def train_sfda(cfg, model_student, model_teacher, args, resume=False):
    model_teacher.eval()
    model_student.train()

    optimizer = build_optimizer(cfg, model_student)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(model_student, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    data_loader = build_detection_train_loader(cfg)

    total_epochs = 10
    len_data_loader = len(data_loader.dataset.dataset.dataset)
    start_iter, max_iter_perepoch = 0, len_data_loader
    max_sf_da_iter = total_epochs*max_iter_perepoch
    logger.info("Starting training from iteration {}".format(start_iter))
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, len_data_loader, max_iter=max_sf_da_iter)
    writers = (
            default_writers(cfg.OUTPUT_DIR, max_sf_da_iter) if comm.is_main_process() else []
        )
    with EventStorage(start_iter) as storage:
        for epoch in range(1, total_epochs+1):
            cfg.defrost()
            cfg.SOURCE_FREE.TYPE = True
            cfg.freeze()
            data_loader = build_detection_train_loader(cfg)
            model_student.train()
            start_iter, max_iter_perepoch = 0, len_data_loader
            progress_bar = tqdm(zip(data_loader, range(start_iter, max_iter_perepoch)))
            for data, iteration in progress_bar:
                storage.iter = iteration+(epoch-1)*max_iter_perepoch
                with torch.no_grad():
                    _, teacher_features, teacher_proposals, teacher_results = model_teacher(data, mode="train", iteration=iteration)
                teacher_pseudo_results, _ = process_pseudo_label(teacher_results, 0.7, "roih", "thresholding") # HPL with confidence thresholding 0.7
                loss_dict = model_student(data, cfg, model_teacher, teacher_features, teacher_proposals, teacher_pseudo_results, mode="train")
                losses = sum(loss_dict.values())

                assert torch.isfinite(losses).all(), loss_dict

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                
                # reduce losses over all instances in all machines
                loss_dict_reduced = {
                    k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
                }
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                
                progress_bar.set_description(
                    f"epoch: {epoch} iter: {iteration}/{max_iter_perepoch} "+''.join([f'{k}: {v.item():.4f}, ' for k,v in loss_dict.items()])
                )
                
                if (
                    cfg.SOURCE_FREE.EMAPERIOD > 0
                    and (storage.iter + 1) % cfg.SOURCE_FREE.EMAPERIOD == 0
                ):
                    new_teacher_dict = update_teacher_model(model_student, model_teacher, keep_rate=cfg.SOURCE_FREE.KEEP_RATE)
                    model_teacher.load_state_dict(new_teacher_dict)
                    
                if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and ((storage.iter + 1) % cfg.TEST.EVAL_PERIOD == 0
                    or storage.iter == max_sf_da_iter - 1)
                ):
                    validation_sfda(cfg, model_student, model_teacher, storage, writers, iteration, epoch)
                    
                elif storage.iter - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration < max_iter_perepoch-1
                ):
                    for writer in writers:
                        writer.write()
                
                periodic_checkpointer.step(iteration)
            
            # validation_sfda(cfg, model_student, model_teacher, storage, writers, iteration, epoch)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    folder_names= []
    folder_names.append(str(cfg.SOURCE_FREE.METHOD))
    folder_names.append(str(cfg.SOLVER.BASE_LR).split('.')[1])
    folder_names.append(str(cfg.SOURCE_FREE.EMAPERIOD))
    folder_names.append(str(cfg.SOURCE_FREE.KEEP_RATE).split('.')[1])
    import datetime
    current_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR+"_" + "_".join(folder_names)+"_"+current_time
    
    cfg.freeze()
    default_setup(cfg, args)

    seed = 1000
    print("SEED: ", seed)
    print("Output_dir: ", cfg.OUTPUT_DIR)
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
    model_student = build_model(cfg)
    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = "teacher_sfda_RCNN"
    cfg.freeze()
    model_teacher = build_model(cfg)
    # model_anchor = build_model(cfg)
    logger.info("Model:\n{}".format(model_student))

    DetectionCheckpointer(model_student, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    DetectionCheckpointer(model_teacher, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    # DetectionCheckpointer(model_anchor, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    logger.info("Trained model has been sucessfully loaded")
    return train_sfda(cfg, model_student, model_teacher, args)


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