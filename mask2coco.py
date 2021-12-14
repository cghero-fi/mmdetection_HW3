import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import matplotlib.pyplot as plt

dataset_train_path = '/content/drive/MyDrive/LYY_HW3/dataset/train/'

CATEGORIES = [
    {
        'id': 1,
        'name': 'Nuclei',
        'supercategory': 'Nuclei',
    },
]
 
def filter_for_jpeg(root, files):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, 'images', f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files
 
def filter_for_annotations(root, files, image_filename):
    file_types = ['*mask*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    # files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
 
def convert_to_coco(ROOT_DIR=dataset_train_path, json_name='train.json'):
    IMAGE_DIR = os.path.join(ROOT_DIR, "images") 
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "masks") #
    coco_output = {
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
 
    image_id = 1
    segmentation_id = 1
    
    # filter for jpeg images
    for root in os.listdir(ROOT_DIR):

        files = list(os.walk(os.path.join(ROOT_DIR, root, 'images')))
        if len(files) < 1:
            continue
        else:
            files = files[0][2]
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(os.path.join(ROOT_DIR, image_filename))
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)
            # filter for associated png annotations
            for root, _, files in os.walk(os.path.join(ROOT_DIR, root, 'masks')):
                annotation_files = filter_for_annotations(root, files, image_filename)
                # go through each associated annotation
                for annotation_filename in annotation_files:
                    # class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                    class_id = 1
                    category_info = {'id': class_id, 'is_crowd': 0}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)
                    
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                    segmentation_id = segmentation_id + 1
            image_id = image_id + 1

    with open('{}/'.format(ROOT_DIR) + json_name, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

convert_to_coco(ROOT_DIR='/content/drive/MyDrive/LYY_HW3/dataset/train/', json_name='train.json')
convert_to_coco(ROOT_DIR='/content/drive/MyDrive/LYY_HW3/dataset/val/', json_name='val.json')