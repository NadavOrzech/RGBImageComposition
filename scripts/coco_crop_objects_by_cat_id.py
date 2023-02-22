import os
import sys
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from skimage.draw import polygon
import argparse


def get_coco_labels_dict(coco: COCO):

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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--annotations_file", type=str, default="/disk2/nadav/COCO/annotations/instances_train2017.json", help="path of COCO annotations file",
)
parser.add_argument(
    "--images_dir", type=str, default="/disk2/nadav/rula/dataset/coco_stuff/train_img", help="path of COCO images directory",
)
parser.add_argument(
    "--output_dir", type=str, default="_objects_and_masks", help="path of output directory",
)
parser.add_argument(
    "--max_obj", type=int, default=100, help="maximum number of objects to generate",
)
parser.add_argument(
    "--max_frames", type=int, default=None, help="maximum number of frames to process",
)
parser.add_argument(
    "--min_crop", type=int, default=1000, help="minimum size of object bbox to crop",
)
parser.add_argument(
    "--label", type=str, default="motorcycle", help="label to crop",
)
opt, unknown = parser.parse_known_args()

annotations_file = opt.annotations_file
images_base_path = opt.images_dir
output_dir = opt.output_dir
maximum_number_of_objects = opt.max_obj
maximum_number_of_frames = opt.max_frames
minimum_crop_size = opt.min_crop

os.makedirs(output_dir, exist_ok=True)


coco = COCO(annotations_file)
coco_labels_dict = get_coco_labels_dict(coco)

selected_categories = [opt.label]
selected_categories_ids = [coco_labels_dict[label] for label in selected_categories]

cats = coco.loadCats(selected_categories_ids)
imgIds = coco.getImgIds(catIds=coco.getCatIds(cats))
if maximum_number_of_frames is None:
    maximum_number_of_frames = len(imgIds)

total_number_of_objects = 0
for ix, img_id in enumerate(imgIds[:maximum_number_of_frames]):

    # Load the image
    filename = coco.loadImgs(img_id)[0]['file_name']
    img_path = os.path.join(
        images_base_path,
        filename
    )

    img = Image.open(img_path).convert("RGB")
    orig_img = np.array(img)

    # Get annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # Crop objects and masks by category
    output_ann_id = 0
    for ann_i, ann in enumerate(annotations):
        if type(ann["segmentation"]) == list:
            if "segmentation" in ann and ann['category_id'] in selected_categories_ids:
                masked_img = np.zeros_like(orig_img)

                for seg_i, seg in enumerate(ann["segmentation"]):
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    rr, cc = polygon(poly[:, 1] - 1, poly[:, 0] - 1)
                    masked_img[rr, cc, :] = 1


                obj = np.where(masked_img, orig_img, masked_img)
                mask = np.where(masked_img, 255, masked_img)

                rows, cols, _ = np.where(mask == 255)
                min_x, max_x = min(cols), max(cols)
                min_y, max_y = min(rows), max(rows)

                obj = obj[min_y:max_y, min_x:max_x, :]
                mask = mask[min_y:max_y, min_x:max_x, :]
                if obj.shape[0]*obj.shape[1]<minimum_crop_size:
                    continue

                Image.fromarray(obj.astype(np.uint8)).save(os.path.join(output_dir, filename.replace(".jpg", f"_obj_{output_ann_id+1}.jpg")))
                Image.fromarray(mask.astype(np.uint8)).save(os.path.join(output_dir, filename.replace(".jpg", f"_mask_{output_ann_id+1}.jpg")))
                output_ann_id+=1
    
    
    total_number_of_objects+=output_ann_id
    if total_number_of_objects >= maximum_number_of_objects:
        break


