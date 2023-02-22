
from pathlib import Path
import glob
import os
import random
from PIL import Image
import numpy as np
import cv2

from utils.json_editing import load_json, save_json
from utils.image_editing import get_image_shape, paste_object_to_image


class BDDDataGenerator():
    def __init__(self, 
                 bdd_labels_file_path: Path,
                 bdd_imgs_dir_path: Path,
                 objects_and_masks_dir_path: Path,
                 class_label: str,
                 seed: int              
                 ) -> None:
        
        self.objs_path_list = [obj_file for obj_file in sorted(glob.glob(os.path.join(objects_and_masks_dir_path,"*_obj_*")))]
        
        self.bdd_imgs_dir_path = bdd_imgs_dir_path
        self.bdd_labels = load_json(bdd_labels_file_path)

        self.prompt_template = "A CLASS on a SCENE at TIMEOFDAY on a WEATHER weather"
        self.label = class_label
        random.seed(seed)

    def sample(self,number_of_samples=1, bg_img_name=None, obj_img_name=None, x=None, y=None, prompt=None):
        curr_bg_img_path : Path
        curr_obj_img_path : Path
        curr_mask_img_path : Path
        
        if bg_img_name is None:
            bg_img_index = random.randint(0, len(self.bdd_labels)-1)
            bg_img_attr = self.bdd_labels[bg_img_index]['attributes']

            while bg_img_attr['weather'] == "undefined" or bg_img_attr['timeofday'] == "undefined" or bg_img_attr['scene'] == "undefined":
                # in order to get maximum information for prompt generation
                bg_img_index = random.randint(0, len(self.bdd_labels)-1)
                bg_img_attr = self.bdd_labels[bg_img_index]['attributes']

            curr_bg_img_path = os.path.join(self.bdd_imgs_dir_path, self.bdd_labels[bg_img_index]['name'])
        else:
            curr_bg_img_path = os.path.join(self.bdd_imgs_dir_path, bg_img_name)
            assert os.path.isfile(curr_bg_img_path)
            
            for bg_img_index, bdd_dict in enumerate(self.bdd_labels):
                # finding the relevant bdd item for the image
                if bdd_dict['name'] == bg_img_name:
                    bg_img_attr=bdd_dict['attributes']
                    break

        if prompt is None:
            replace_dict = {
                "CLASS": self.label,
                "SCENE": bg_img_attr['scene'],
                "WEATHER": bg_img_attr['weather'],
                "TIMEOFDAY": bg_img_attr['timeofday']
            }
            prompt = self.prompt_template
            for k,v in replace_dict.items():
                prompt = prompt.replace(k, v)

        if obj_img_name is None:
            obj_img_index = random.randint(0, len(self.objs_path_list)-1)
            curr_obj_img_path = self.objs_path_list[obj_img_index]
            curr_mask_img_path = curr_obj_img_path.replace("_obj_", "_mask_")
        else:
            curr_obj_img_path = os.path.join(
                os.path.dirname(self.objs_path_list[0]),
                obj_img_name
            )
            curr_mask_img_path = curr_obj_img_path.replace("_obj_", "_mask_")
            assert os.path.isfile(curr_obj_img_path)


        composed_img, composed_mask, bbox = self.composite_object(
                                            background_img_path=curr_bg_img_path,
                                            object_img_path=curr_obj_img_path,
                                            mask_img_path=curr_mask_img_path,
                                            x=x,y=y,
                                            number_of_samples=number_of_samples,
                                            bdd_img_dict=self.bdd_labels[bg_img_index],
                                        )
        
        return composed_img, composed_mask, [prompt]*number_of_samples, bbox, (curr_bg_img_path, curr_obj_img_path), self.bdd_labels[bg_img_index]


    def composite_object(
                        self,
                        background_img_path,
                        object_img_path,
                        mask_img_path,
                        x,y,
                        number_of_samples,
                        bdd_img_dict=None,
                        ): 

        if x is not None and y is not None:
            # for manualy placed objects
            output_image, output_mask, bbox = paste_object_to_image(obj_path=object_img_path, background_path=background_img_path, 
                                mask_path=mask_img_path, output_path=None, x=x, y=y)
            
            return [Image.fromarray(output_image.astype(np.uint8))], [Image.fromarray(output_mask.astype(np.uint8))], [bbox]


        obj_shape = get_image_shape(object_img_path)
        composed_imgs, composed_masks, coordinates = [], [], []
        for i in range(number_of_samples):
            x,y, drivable_area  = self._get_coordinates_by_segmentation_map(bdd_img_dict,obj_shape)

            output_image, output_mask, bbox = paste_object_to_image(obj_path=object_img_path, background_path=background_img_path, 
                                    mask_path=mask_img_path, output_path=None, x=x, y=y)

            composed_imgs.append(Image.fromarray(output_image.astype(np.uint8)))
            composed_masks.append(Image.fromarray(output_mask.astype(np.uint8)))
            coordinates.append(bbox)
        
        return composed_imgs, composed_masks, coordinates


    def _get_coordinates_by_segmentation_map(self, background_img_dict, obj_shape, shape=(720,1280)):
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for label in background_img_dict['labels']:
            if label['category'] == 'drivable area':
                points = np.array(label['poly2d'][0]['vertices'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [points], 1)
            # TODO: Check if all driveable area is relevant ?

        nonzero_y, nonzero_x = np.nonzero(mask)
        if len(nonzero_y) == 0:
            return None
        random_index = np.random.choice(len(nonzero_y))

        return nonzero_x[random_index]-(obj_shape[0]//2), nonzero_y[random_index]-(obj_shape[1]), mask


