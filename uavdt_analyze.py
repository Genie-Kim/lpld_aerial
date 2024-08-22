import os
import json
import csv

# Define the path to the metadata folder
meta_folder = '/hddsda/lpld_dataset/visdroneDET/visdrone2019-det/train/ann'
# meta_folder = '/hddsda/lpld_dataset/uavdt_temp/uavdt/test/ann'

overall_metadatas = "/hddsda/lpld_dataset/visdroneDET/visdrone2019-det/meta.json"

attrs_id2name={}
with open(overall_metadatas, 'r') as file:
    overall_metadata = json.load(file)
    taglist = overall_metadata['tags']
    for tag in taglist:
        attrs_id2name[tag['id']] = tag['name']
    
# Initialize the dataset dictionary
dataset = {}
havecar_data={}
# Iterate over all JSON files in the metadata folder
for filename in os.listdir(meta_folder):
    if filename.endswith('.json'):
        # Construct the full path to the JSON file
        filepath = os.path.join(meta_folder, filename)                
        # Read the JSON file
        with open(filepath, 'r') as file:
            taglist = json.load(file)['tags']
            attrids_image=[]
            for tag in taglist:
                attrids_image.append(tag['tagId'])
        
        with open(filepath, 'r') as file:
            docu = file.read()
            findstring = '"classTitle": "car"'
            havecar = 1 if findstring in docu else 0
                
        
        # Extract the image name (without extension) to use as a key
        image_name=filename.split('.')[0]
        
        # Add the image and its attributes to the dataset dictionary
        dataset[image_name] = attrids_image
        havecar_data[image_name] = havecar
        

# Prepare data for CSV
csv_data = []

# Create a header row with the attribute list
header = ["image"] + list(attrs_id2name.values())
header = header + ["havecar"]
csv_data.append(header)

# Fill in the rows
for image, attrids_image in dataset.items():
    row = [image]  # Start with the image name
    for attrid in list(attrs_id2name.keys()):
        row.append(1 if attrid in attrids_image else 0)  # 1 if attribute present, otherwise 0
    row.append(havecar_data[image])
    csv_data.append(row)

# Write the data to a CSV file
output_csv = 'image_attributes.csv'
with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"CSV file '{output_csv}' has been created successfully.")
