import argparse


def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bdd_labels_file",
        type=str,
        default="/disk2/nadav/bdd100k/labels/bdd100k_labels_images_train.json",
        help="path of BDD100k labels file",
    )

    parser.add_argument(
        "--bdd_images_dir",
        type=str,
        default="/disk2/nadav/bdd100k/images/100k/train",
        help="path of BDD100k images directory",
    )

    parser.add_argument(
        "--objects_images_dir",
        type=str,
        default="/disk2/nadav/__temp_results_/",
        help="path of objects and masks images directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="path of ILVR results directory",
    )

    parser.add_argument(
        "--samples_per_bdd_image",
        type=int,
        default=1,
        help="number of automatically generated samples per bdd image",
    )


    parser.add_argument(
        "--num_of_images",
        type=int,
        default=10,
        help="number of different images from bdd dataset",
    )

    parser.add_argument(
        "--class_name",
        type=str,
        default="bus",
        help="name of objects class (required for automatically generated prompts)",
    )

    parser.add_argument(
        "--bg_img_name",
        type=str,
        default=None,
        help="background image name (optional)",
    )

    parser.add_argument(
        "--obj_img_name",
        type=str,
        default=None,
        help="object image name (optional)",
    )

    parser.add_argument(
        "--x",
        type=int,
        default=None,
        help="horizontal position of object (optional)",
    )

    parser.add_argument(
        "--y",
        type=int,
        default=None,
        help="vertical position of object (optional)",
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="the prompt to render"
    )

            
    return parser

