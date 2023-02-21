import os
import datetime

from utils.json_editing import save_json

class ResultsOrginizer():
    def __init__(self, output_dir) -> None:
        os.makedirs(output_dir, exist_ok=True)  

        time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.output_dir = os.path.join(output_dir, time)
        os.makedirs(self.output_dir)

    
    def save_results(self, bg_img_path, object_path, orig_img_list, results_list, coordinates, prompts):
        output_dir_name = os.path.basename(bg_img_path).split(".jpg")[0] + "_" + os.path.basename(object_path).split(".jpg")[0]
        curr_output_dir = os.path.join(self.output_dir, output_dir_name)
        os.makedirs(curr_output_dir, exist_ok=True)

        output_data = {
            "orig_background_image_path": bg_img_path,
            "orig_object_image_path": object_path,
            "orig_mask_image_path": object_path.replace("_obj_", "_mask_")
        }
        for i, (orig_img, res) in enumerate(zip(orig_img_list, results_list)):
            orig_img.save(os.path.join(curr_output_dir, f'img_sample_{i}.png'))
            res.save(os.path.join(curr_output_dir, f'res_sample_{i}.png'))

            output_data[f"sample_{i}"] = {
                "coordinates": str((coordinates[i][1], coordinates[i][0])),
                "prompt": prompts[i]
            }

        save_json(os.path.join(curr_output_dir, "info.json"), output_data)
    
    def create_coco_dict(self, bdd_labels_dict):
        coco_image = {
            "id": 0,
            "width": image['width'],
            "height": image['height'],
            "file_name": image['name'],
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        coco_labels['images'].append(coco_image)
        image_id += 1