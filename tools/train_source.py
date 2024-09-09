#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    MetadataCatalog,
    DatasetMapper,
)
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    print_csv_format,
    SemSegEvaluator,
    CityscapeCarDetectionEvaluator,
    ClipartDetectionEvaluator,
    WatercolorDetectionEvaluator,
    CityscapeDetectionEvaluator,
    FoggyDetectionEvaluator,
    KaistDetectionEvaluator,
    KaistPersonDetectionEvaluator,
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
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

logger = logging.getLogger("detectron2")


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
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
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
    if evaluator_type == "kaist_viz":
        return KaistDetectionEvaluator(dataset_name)
    if evaluator_type == "kaist_tr":
        return KaistDetectionEvaluator(dataset_name)
    if evaluator_type == "kaist_viz_person":
        return KaistPersonDetectionEvaluator(dataset_name)
    if evaluator_type == "kaist_tr_person":
        return KaistPersonDetectionEvaluator(dataset_name)
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
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        cfg.defrost()
        cfg.SOURCE_FREE.TYPE = False
        cfg.freeze()
        # Add other cfg setup code here
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


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    # scheduler = build_lr_scheduler(cfg, optimizer)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=cfg.SOLVER.PATIENCE, eps=1e-8)
    # get_last_lr()
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))

    best_mAP = 0
    with EventStorage(start_iter) as storage:
        progress_bar = tqdm(zip(data_loader, range(start_iter, max_iter)))
        for data, iteration in progress_bar:
            storage.iter = iteration
            loss_dict = model(data)
            try:
                losses = sum(loss_dict.values())
            except:
                continue
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
            )
            # scheduler.step()
            
            progress_bar.set_description(
                    f"[{iteration}/{max_iter}] "+''.join([f'{k}: {v.item():.4f}, ' for k,v in loss_dict.items()])
                )

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                results = do_test(cfg, model)
                mAP = results['bbox']['AP50']
                scheduler.step(mAP)
                class_ap50 = results['bbox'].pop('class-AP50')
                # scheduler.step(class_ap50[0])
                class_ap50 = {f'class-AP50/{k}': v for k,v in enumerate(class_ap50)}
                storage.put_scalars(**results['bbox'])
                storage.put_scalars(**class_ap50)
                for writer in writers:
                    writer.write()
                
                if mAP > best_mAP:
                    best_mAP = mAP
                    checkpointer.save("best_mAP")
                    logger.info(f'Save Best Score: {best_mAP:.4f}')
                    
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "model_{}.pth".format(iteration)))
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

                
            elif iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


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
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


def invoke_main() -> None:
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

if __name__ == "__main__":
    invoke_main()  # pragma: no cover