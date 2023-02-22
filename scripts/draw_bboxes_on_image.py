import cv2
import json
import glob
import os

def draw_bboxes_on_image(image_path, labels_dict):
    # Load image
    image = cv2.imread(image_path)
    # Extract labels
    labels = labels_dict['labels']
    # Draw bounding boxes on image
    for label in labels:
        if 'box2d' not in label: continue
        bbox = label['box2d']
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        category = label['category']
        color = get_color_for_category(category)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, category, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    # Show image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def get_color_for_category(category):
    # Assign a color to each category
    if category == "car":
        return (0, 255, 0)  # Green
    elif category == "person":
        return (0, 0, 255)  # Red
    elif category == "traffic sign":
        return (255, 0, 0)  # Blue
    else:
        return (255, 255, 255)  # White (for other categories)


base_folder = "/disk2/nadav/source/RGBImageComposition/results/2023-02-22_10-59-53"
for subdir in glob.glob(os.path.join(base_folder, '*')):
    json_file = os.path.join(subdir, "bdd_0.json")
    img_path = os.path.join(subdir, "img_sample_0.png")
    with open(json_file, 'r') as f:
        data = json.load(f)
    draw_bboxes_on_image(img_path, data)

