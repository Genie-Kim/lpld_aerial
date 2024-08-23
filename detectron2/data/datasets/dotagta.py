import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_dotagta_instances", "register_dotagta"]

# CLASS_NAMES = ('plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
#                'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle',
#                'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool')
CLASS_NAMES = ("car", "background")
mapper = {'large-vehicle': 'car', 'small-vehicle': 'car'}

def load_dotagta_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local

    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".png")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            
            if cls in mapper:
                cls = mapper[cls]
                if cls in class_names:
                    instances.append(
                        {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                    )

        r["annotations"] = instances
        dicts.append(r)
    return dicts

def register_dotagta(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_dotagta_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
