# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
import os
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
import cv2
import numpy as np

import xml.etree.ElementTree as ET
from detectron2.utils.file_io import PathManager
import pdb

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.structures import pairwise_iou, Boxes
import matplotlib.pyplot as plt

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects

def inference_on_dataset(
    model, data_loader, evaluator, draw=False, dirname="", return_tpfn=False):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    
    if return_tpfn:
        tp_height, tp_width = [], []
        fn_height, fn_width = [], []
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if draw:            
                sample = cv2.imread(inputs[0]["file_name"])
                bbox_per_sample = []
                for bbox in outputs[0]["instances"].pred_boxes.tensor.cpu().numpy():
                    bbox_per_sample.append(bbox)
                
                for bbox in bbox_per_sample:
                    cv2.rectangle(sample, (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2) # red color
                
                annopath = evaluator._anno_file_template
                classnames = evaluator._class_names
                img_id = inputs[0]["image_id"]
                
                annotations = parse_rec(annopath.format(img_id))
                gt_boxes = np.array([x["bbox"] for x in annotations if x["name"] in classnames])
                
                if len(gt_boxes) > 0:
                    for bbox in range(len(gt_boxes)):
                        cv2.rectangle(sample, (gt_boxes[bbox][0], gt_boxes[bbox][1]),
                                    (gt_boxes[bbox][2], gt_boxes[bbox][3]), (0, 255, 0), 2) # green color
                    
                    cv2.imwrite(os.path.join("./visualization", dirname, f"{img_id}.png"), sample)
                    
            if return_tpfn:
                predictions = outputs[0]["instances"].pred_boxes.tensor.cpu()
                pred_classes = outputs[0]["instances"].pred_classes.cpu()     
                annopath = evaluator._anno_file_template
                classnames = evaluator._class_names
                img_id = inputs[0]["image_id"]
                
                annotations = parse_rec(annopath.format(img_id))
                gt_boxes = np.array([x["bbox"] for x in annotations if x["name"] in classnames])
                gt_classes = [classnames.index(x["name"]) for x in annotations if x["name"] in classnames]
                
                iou_results = pairwise_iou(Boxes(gt_boxes), Boxes(predictions))

                if len(predictions) > 0:
                    for i in range(len(iou_results)):
                        max_value, max_index = iou_results[i].max(dim=0)
                        if max_value > 0.5 and gt_classes[i] == pred_classes[max_index]:
                            tp_height.append(float(gt_boxes[i][3] - gt_boxes[i][1]))
                            tp_width.append(float(gt_boxes[i][2] - gt_boxes[i][0]))
                        elif max_value == 0:
                            fn_height.append(float(gt_boxes[i][3] - gt_boxes[i][1]))
                            fn_width.append(float(gt_boxes[i][2] - gt_boxes[i][0]))
                            
                    plt.figure(figsize=(15, 8))
                    plt.scatter(tp_width, tp_height, color='mediumseagreen', label=f'TP : {len(tp_height)}')
                    plt.scatter(fn_width, fn_height, color='darkorange', label=f'FN : {len(fn_height)}')
                    plt.xlabel('Width')
                    plt.ylabel('Height')
                    plt.legend(fontsize=15)
                    plt.savefig("tpfn.png")
                    plt.close()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    # logger.info(
    #     "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_time_str, total_time / (total - num_warmup), num_devices
    #     )
    # )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
    #     )
    # )
    
    ###### results = OrderedDict([('bbox', {'AP': 0.454, 'AP50': 0.454, 'AP75': 0.454})])
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def inference_on_corruption_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            base_dict = {k: v for k, v in inputs[0].items() if "image" not in k}
            base_dict["image_id"] = inputs[0]["image_id"]
            for severity in range(4,5):
                corrupt_inputs = base_dict.copy()
                corrupt_inputs["image"] = inputs[0]["image_" + str(severity)]
                corrupt_inputs = [corrupt_inputs]
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(corrupt_inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(corrupt_inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                #     eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                #     log_every_n_seconds(
                #         logging.INFO,
                #         (
                #             f"Inference done {idx + 1}/{total}. "
                #             f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                #             f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                #             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                #             f"Total: {total_seconds_per_iter:.4f} s/iter. "
                #             f"ETA={eta}"
                #         ),
                #         n=5,
                #     )
                start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    # logger.info(
    #     "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_time_str, total_time / (total - num_warmup), num_devices
    #     )
    # )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
    #     )
    # )
    
    ###### results = OrderedDict([('bbox', {'AP': 0.454, 'AP50': 0.454, 'AP75': 0.454})])
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def inference_on_cropped_dataset(model, data_loader, evaluator, crop_size, overlap):
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    
    results = {}
    total = 0
    evaluator.reset()
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        for inputs in data_loader:    
            image = inputs[0]["image"]
            image_height, image_width = image.shape[-2:]
            pad_h = max(crop_size - image_height, 0)
            pad_w = max(crop_size - image_width, 0)
            
            if pad_h > 0 or pad_w > 0:
                image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), value=0)

            h, w = image.shape[-2:]
            results = OrderedDict()
            results["bbox"] = {}
            for y in range(0, h - crop_size + 1, crop_size - overlap):
                for x in range(0, w - crop_size + 1, crop_size - overlap):
                    total += 1
                    crop = image[..., y:y+crop_size, x:x+crop_size]
                    crop_inputs = inputs.copy()
                    crop_inputs[0]["image"] = crop
                    outputs = model(crop_inputs)
                    evaluator.process(crop_inputs, outputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
                    ###### results = OrderedDict([('bbox', {'AP': 0.454, 'AP50': 0.454, 'AP75': 0.454})])
                    results_tmp = evaluator.evaluate_crop(x, y, crop_size) 
                    for metric, value in results_tmp["bbox"].items():
                        if metric not in results["bbox"]:
                            results["bbox"][metric] = value
                        else:
                            if isinstance(value, float):
                                results["bbox"][metric] += value
                            else:
                                for idx, v in enumerate(value):                                
                                    results["bbox"][metric][idx] += v
                    total += 1
                    evaluator.reset()
                    
            print(results)

    for metric, value in results:
        if isinstance(value, float):
            results[metric] = value / total
        else:
            for idx, v in value:
                results[metric][idx] = v / total

    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
