import os
# import sys
# sys.path.append("CS2Real")
# sys.path.append("ResizeRight") # TODO

from utils.json_editing import *
from utils.image_editing import *
from modules.ILVRSampler import ILVRSampler
from modules.BDDDataGenerator import BDDDataGenerator
from modules.ResultsOrginizer import ResultsOrginizer
from config.ILVR_options import create_ILVR_argparser
from config.base_options import create_argparser

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def main():
    ilvr_parser = create_ILVR_argparser()
    ilvr_opt, unknown = ilvr_parser.parse_known_args()

    base_parser = create_argparser()
    opt, unknown = base_parser.parse_known_args()

    bdd_generator = BDDDataGenerator(
    opt.bdd_labels_file,
    opt.bdd_images_dir,
    opt.objects_images_dir,
    opt.class_name
    )
    sampler = ILVRSampler(ilvr_opt)
    res_orginzier = ResultsOrginizer(opt.output_dir)
    
    if opt.bg_img_name is not None \
        or opt.obj_img_name is not None \
            or opt.x is not None \
                or opt.y is not None:
        opt.num_of_images = 1

    for kk in range(opt.num_of_images):
        img, mask, prompt, bbox, paths, bdd_labels = bdd_generator.sample(
            number_of_samples=opt.samples_per_bdd_image, 
            bg_img_name=opt.bg_img_name,
            obj_img_name=opt.obj_img_name,
            x=opt.x,
            y=opt.y,
            prompt=opt.prompt
            )
        

        result = sampler.sample(img,mask,prompt)
        
        res_orginzier.save_results(paths[0], paths[1], img, result, bbox, prompt, bdd_labels, opt.class_name)



if __name__=="__main__":
    main()