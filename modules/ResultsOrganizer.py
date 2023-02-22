import os
import datetime

from utils.json_editing import save_json

class ResultsOrganizer():
    def __init__(self, output_dir) -> None:
        os.makedirs(output_dir, exist_ok=True)  

        time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.output_dir = os.path.join(output_dir, time)
        os.makedirs(self.output_dir)

    
    def save_results(self, bg_img_path, object_path, orig_img_list, results_list, bbox, prompts, bdd_labels, new_category):
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
                "coordinates": str(bbox),
                "prompt": prompts[i]
            }

            new_labels = self.add_bbox_to_bdd(labels_dict=bdd_labels,new_bbox=bbox[i], new_category=new_category)
            
            save_json(os.path.join(curr_output_dir, f"bdd_ref_{i}.json"), new_labels)
        save_json(os.path.join(curr_output_dir, f"bdd_orig.json"), bdd_labels)
        
        save_json(os.path.join(curr_output_dir, "info.json"), output_data)


    def add_bbox_to_bdd(self, labels_dict, new_bbox, new_category):
        # Extract existing labels
        labels = labels_dict['labels']
        
        indices_to_remove = []

        # Adjust coordinates of existing bboxes
        for l_i, label in enumerate(labels):
            if 'box2d' not in label: continue
            bbox = label['box2d']
            bbox = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
            if self.bbox_intersects(bbox, new_bbox):
                modified_bbox = self.adjust_bbox(bbox, new_bbox)
                if modified_bbox is None:
                    # new bbox containes the previous one
                    indices_to_remove.append(l_i)
                    continue
                
                
                if ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))<=0.1*((modified_bbox[2]-modified_bbox[0])*(modified_bbox[3]-modified_bbox[1])):
                    # the remains of the previous bbox is too small, removing
                    indices_to_remove.append(l_i)
                    continue

                labels[l_i]['box2d']['x1'] = int(modified_bbox[0])
                labels[l_i]['box2d']['y1'] = int(modified_bbox[1])
                labels[l_i]['box2d']['x2'] = int(modified_bbox[2])
                labels[l_i]['box2d']['y2'] = int(modified_bbox[3])

        indices_to_remove.sort(reverse=True)
        for l_i in indices_to_remove:
            labels.pop(l_i)


        # Append new label
        new_label = {
            "category": new_category,
            "attributes": {},
            "manualAttributes": {},
            "box2d": {
                "x1": int(new_bbox[0]),
                "y1": int(new_bbox[1]),
                "x2": int(new_bbox[2]),
                "y2": int(new_bbox[3])
            }
        }
        labels.append(new_label)
        # Update dictionary
        labels_dict['labels'] = labels
        return labels_dict

    def bbox_intersects(self, bbox1, bbox2):
        # Check if two bboxes intersect
        x1, y1, x2, y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        x3, y3, x4, y4 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
        return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

    # def adjust_bbox1(self, bbox1, bbox2):
    #     # Adjust coordinates of bbox1 so that it doesn't intersect with bbox2
    #     x1, y1, x2, y2 = bbox1['x1'], bbox1['y1'], bbox1['x2'], bbox1['y2']
    #     x3, y3, x4, y4 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    #     if x1 < x3 and x2 < x4:
    #         x2 = x3
    #     elif x1 > x3 and x2 > x4:
    #         x1 = x4
    #     elif x1 < x3 and x2 > x4:
    #         x1, x2 = x3, x4
    #     elif x1 > x3 and x2 < x4:
    #         x1, x2 = x3, x4

    #     if y1 < y3 and y2 < y4:
    #         y2 = y3
    #     elif y1 > y3 and y2 > y4:
    #         y1 = y4
    #     elif y1 < y3 and y2 > y4:
    #         y1, y2 = y3, y4
    #     elif y1 > y3 and y2 < y4:
    #         y1, y2 = y3, y4

    #     return [x1, y1, x2, y2]


    def adjust_bbox(self, bbox, new_bbox):
        # for i, bbox in enumerate(bboxes):
        #     if bbox[0] < new_bbox[2] and bbox[2] > new_bbox[0] and bbox[1] < new_bbox[3] and bbox[3] > new_bbox[1]:
                # the current bbox intersects with the new one
        if bbox[0] > new_bbox[0] and bbox[1] > new_bbox[1] and bbox[2] < new_bbox[2] and bbox[3] < new_bbox[3]:
            # new bbox containes the previous one
            return None


        if bbox[0] < new_bbox[0]:
            bbox = (bbox[0], bbox[1], new_bbox[0], bbox[3])
        elif bbox[2] > new_bbox[2]:
            bbox = (new_bbox[2], bbox[1], bbox[2], bbox[3])
        elif bbox[1] < new_bbox[1]:
            bbox = (bbox[0], bbox[1], bbox[2], new_bbox[1])
        elif bbox[3] > new_bbox[3]:
            bbox = (bbox[0], new_bbox[3], bbox[2], bbox[3])
        # bboxes[i] = bbox
        return bbox
