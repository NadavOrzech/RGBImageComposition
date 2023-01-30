import os
import sys
import json
import numpy as np
from pycocotools.coco import COCO

# Load the annotations file
# annotations_file = '/disk2/nadav/COCO/annotations/instances_train2017.json'

def get_coco_labels_dict(coco: COCO):
    # coco = COCO(annotations_file)

    # Get all categories
    cat_ids = coco.getCatIds()
    categories = coco.loadCats(cat_ids)

    # Initialize a dictionary to store the labels and their indices
    labels_dict = {}

    # Iterate over each category
    for category in categories:
        labels_dict[category['name']] = category['id']

    # Print the dictionary
    return labels_dict