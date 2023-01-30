import os
import sys
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from skimage.draw import polygon

from get_coco_labels_dict import get_coco_labels_dict

# Load the annotations file
annotations_file = '/disk2/nadav/COCO/annotations/instances_train2017.json'
images_base_path = "/disk2/nadav/rula/dataset/coco_stuff/train_img"
output_dir = "/disk2/nadav/__temp_results"
maximum_number_of_objects = 100
maximum_number_of_frames = None
minimum_crop_size = 1000
os.makedirs(output_dir, exist_ok=True)


coco = COCO(annotations_file)
coco_labels_dict = get_coco_labels_dict(coco)

selected_categories = [
    'bus'
]
selected_categories_ids = [coco_labels_dict[label] for label in selected_categories]

cats = coco.loadCats(selected_categories_ids)
imgIds = coco.getImgIds(catIds=coco.getCatIds(cats))
total_number_of_objects = 0
for ix, img_id in enumerate(imgIds):

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
                    masked_img[rr, cc, :] = 1#count


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
    
    aa=3
    
    total_number_of_objects+=output_ann_id
    if total_number_of_objects >= maximum_number_of_objects:
        exit(0)


