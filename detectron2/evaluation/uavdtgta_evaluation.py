# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator

class UAVDTGtaDetectionEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, source_dataset_name=""):
        self._dataset_name = dataset_name
        self._source_dataset_name = source_dataset_name
        meta = MetadataCatalog.get(dataset_name)

        # Too many tiny files, download all to local for speed.
        annotation_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        self._class_mapper = meta.class_mapper
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            recs = defaultdict(list)
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                thresh=50
                rec, prec, ap = voc_eval(
                    res_file_template,
                    self._anno_file_template,
                    self._image_set_path,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007,
                    mapper=self._class_mapper
                )
                aps[thresh].append(ap * 100)
                if "rec_small" in rec:
                    recs["small"].append(rec["rec_small"])
                if "rec_medium" in rec:
                    recs["medium"].append(rec["rec_medium"])
                if "rec_large" in rec:
                    recs["large"].append(rec["rec_large"])

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        mReC = {diff: np.mean(x) for diff, x in recs.items()}

        ret["bbox"] = {"AP": np.mean(list(mAP.values())),
                       "AP50": mAP[50], "class-AP50": aps[50],
                       "FNR_small" : 100-100*mReC["small"] if "small" in mReC else 100,
                       "FNR_medium" : 100-100*mReC["medium"] if "medium" in mReC else 100,
                       "FNR_large" : 100-100*mReC["large"] if "large" in mReC else 100}

        return ret

@lru_cache(maxsize=None)
def parse_rec(filename,mapper_items=None):
    """Parse a PASCAL VOC xml file."""
    # Reconstruct the mapper dictionary
    mapper = dict(mapper_items) if mapper_items is not None else None
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        if mapper is not None and obj_struct["name"] in mapper.keys():
            obj_struct["name"] = mapper[obj_struct["name"]]
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


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False,mapper=None):
    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    mapper = tuple(sorted(mapper.items())) if mapper is not None else None
    
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename),mapper)

    # thresholds for categorizing object sizes (in terms of area)
    small_thresh = 32 * 32
    medium_thresh = 96 * 96

    # extract gt objects for this class
    class_recs = {}
    npos = 0  # Total number of positive samples (not difficult)
    npos_small = 0
    npos_medium = 0
    npos_large = 0

    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        
        npos += sum(~difficult)
        
        for i, box in enumerate(bbox):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if not difficult[i]:
                if area < small_thresh:
                    npos_small += 1
                elif area < medium_thresh:
                    npos_medium += 1
                else:
                    npos_large += 1

        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    tp_small = np.zeros(nd)
    tp_medium = np.zeros(nd)
    tp_large = np.zeros(nd)
    fp_small = np.zeros(nd)
    fp_medium = np.zeros(nd)
    fp_large = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0  # Mark as true positive
                    area = (BBGT[jmax, 2] - BBGT[jmax, 0]) * (BBGT[jmax, 3] - BBGT[jmax, 1])
                    if area < small_thresh:
                        tp_small[d] = 1.0
                    elif area < medium_thresh:
                        tp_medium[d] = 1.0
                    else:
                        tp_large[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0  # False positive (duplicate detection)
                    area = (BBGT[jmax, 2] - BBGT[jmax, 0]) * (BBGT[jmax, 3] - BBGT[jmax, 1])
                    if area < small_thresh:
                        fp_small[d] = 1.0
                    elif area < medium_thresh:
                        fp_medium[d] = 1.0
                    else:
                        fp_large[d] = 1.0
        else:
            fp[d] = 1.0  # False positive (no matching ground truth)
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if area < small_thresh:
                fp_small[d] = 1.0
            elif area < medium_thresh:
                fp_medium[d] = 1.0
            else:
                fp_large[d] = 1.0

    # compute precision recall for small, medium, and large objects
    def compute_rec_prec(tp, fp, npos):
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        return rec, prec

    # Calculate precision and recall for each size category
    rec_small, _ = compute_rec_prec(tp_small, fp_small, npos_small)
    rec_medium, _ = compute_rec_prec(tp_medium, fp_medium, npos_medium)
    rec_large, _ = compute_rec_prec(tp_large, fp_large, npos_large)

    # Compute overall precision and recall
    rec, prec = compute_rec_prec(tp, fp, npos)
    
    # compute AP (average precision)
    ap = voc_ap(rec, prec, use_07_metric)
    recall = {}
    if npos_small > 0 and len(rec_small):
        recall["rec_small"] = rec_small[-1]
    if npos_medium > 0 and len(rec_medium):
        recall["rec_medium"] = rec_medium[-1]
    if npos_large > 0 and len(rec_large):
        recall["rec_large"] = rec_large[-1]
        
    return recall, prec, ap
