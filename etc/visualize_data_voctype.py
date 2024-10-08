import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Function to parse the XML file
def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append((name, (xmin, ymin, xmax, ymax)))
    
    return boxes

# Function to visualize the annotations
def visualize_annotations(image_path, xml_file):
    # Load the image
    image = Image.open(image_path)

    # Parse the XML file
    boxes = parse_voc_annotation(xml_file)

    # Draw the bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for name, (xmin, ymin, xmax, ymax) in boxes:
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), name, fill="red")

    # Show the image with annotations
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example usage
import os
dataset='uavdt_voc'
ext = 'jpg'
image_id = 'M0701_img001143'

datasetpath = os.path.join('dataset', dataset)
image_path = os.path.join(datasetpath, 'JPEGImages', f'{image_id}.{ext}')
xml_file = os.path.join(datasetpath, 'Annotations', f'{image_id}.xml')
visualize_annotations(image_path, xml_file)
