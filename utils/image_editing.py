

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_image_shape(img_path):
    img = load_pil_image(img_path)
    return img.size

def load_pil_image(img_path, image_size=None,image_resize_factor=None):
    image = Image.open(img_path).convert("RGB")
    if image_size:
        w,h = image_size
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    
    if image_resize_factor:
        w, h = image.size
        w, h = int(w * image_resize_factor), int(h * image_resize_factor)
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)


    return image
    
    
def load_np_image(img_path, image_size=None,image_resize_factor=None):
    image = load_pil_image(img_path, image_size, image_resize_factor)
    image = np.array(image).astype(np.float32) #/ 255.0
    return image


def paste_object_to_image(obj_path, background_path, mask_path, output_path, x=0,y=0, image_resize_factor=None):
    # small_object_size = (256)
    img = load_np_image(obj_path, image_resize_factor=image_resize_factor)
    mask = load_np_image(mask_path, image_resize_factor=image_resize_factor)
    mask[mask>128] = 255
    mask[mask<=128] = 0
    inpainted_img = load_np_image(background_path)
    
    resized_img_shape = img.shape

    # keep object in image boundaries
    y = max(y,0)
    x = max(x,0)
    y = min(y, inpainted_img.shape[0] - resized_img_shape[0])
    x = min(x, inpainted_img.shape[1] - resized_img_shape[1])

    resized_mask = np.zeros_like(inpainted_img)
    resized_mask[y:y+resized_img_shape[0], x:x+resized_img_shape[1], :] = mask
    resized_img = np.zeros_like(inpainted_img)
    resized_img[y:y+resized_img_shape[0], x:x+resized_img_shape[1], :] = img
    
    composed_image = np.where(resized_mask, resized_img, inpainted_img)

    # object_center = (x+(resized_img_shape[1]//2), y+(resized_img_shape[0]//2))

    return composed_image, resized_mask, (x,y)
