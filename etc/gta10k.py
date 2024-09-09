import xml.etree.ElementTree as ET
import os
base_dir = "/hddsda/lpld_dataset/GTAV10K/Annotations"
for xmlfile in os.listdir(base_dir):
    # Load and parse the XML file
    tree = ET.parse(os.path.join(base_dir, xmlfile))
    root = tree.getroot()

    dataset_info=set()
    
    # Iterate over the child elements
    for child in root:
        if child.tag == 'filename':
            filename = child.text
            fileid = filename.split('.')[0]
        elif child.tag == 'size':
            for subchild in child:
                if subchild.tag == 'width':
                    width = subchild.text
                if subchild.tag == 'height':
                    height = subchild.text
        elif child.tag == 'object':
            for subchild in child:
                if subchild.tag == 'name':
                    category = subchild.text
                    dataset_info.add(category)
                    
                if subchild.tag == 'bndbox':
                    for subsubchild in subchild:
                        if subsubchild.tag == 'xmin':
                            xmin = subsubchild.text
                        if subsubchild.tag == 'ymin':
                            ymin = subsubchild.text
                        if subsubchild.tag == 'xmax':
                            xmax = subsubchild.text
                        if subsubchild.tag == 'ymax':
                            ymax = subsubchild.text
        else:
            continue
    
    print(1)
        
        

