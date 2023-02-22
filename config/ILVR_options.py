import argparse


def create_ILVR_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_size",
        type=int,
        nargs="?",
        default=512,
        help="image resolution (will resize image)",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        nargs="*",
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        nargs='*',
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given image",
    )
       
    parser.add_argument(
        "--scale",
        type=float,
        nargs='*',
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength_in",
        type=float,
        nargs='*',
        default=1.0,
        help="strength for noising/unnoising inside mask. 1.0 corresponds to full destruction of information in init image",
    )
    
    parser.add_argument(
        "--strength_out",
        type=float,
        nargs='*',
        default=None,
        help="strength for noising/unnoising outside mask. 1.0 corresponds to full destruction of information in init image",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="/disk2/nadav/source/CS2Real/configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/disk2/nadav/source/CS2Real/models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        nargs='*',
        default=0,
        help="the seed (for reproducible sampling)",
    )
    
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--down_N_in",
        type=int,
        nargs='*',
        default=1,
        help="ILVR downsampling factor inside mask"
    )

    parser.add_argument(
        "--down_N_out",
        type=int,
        nargs='*',
        default=1,
        help="ILVR downsampling factor outside mask"
    )

    parser.add_argument(
        "--T_out",
        type=float,
        nargs='*',
        default=1.0,
        help="strength of ILVR outside mask (in [0.0, 0.1])"
    )

    parser.add_argument(
        "--T_in",
        type=float,
        nargs='*',
        default=0.6,
        help="strength of ILVR inside mask (in [0.0, 0.1])"
    )

    parser.add_argument(
        "--blend_pix",
        type=int,
        nargs='*',
        default=0,
        help="number of pixels for mask smoothing"
    )

    parser.add_argument(
        "--pixel_cond_space",
        action='store_true',
        help="if enabled, uses pixel space for ILVR conditioning, otherwise uses latent space",
    )

    parser.add_argument(
        "--repaint_start",
        type=float,
        nargs='*',
        default=0.6,
        help="Use RePaint (https://arxiv.org/pdf/2201.09865.pdf) for conditioning for (r*time_steps steps), (r in [0.0, 0.1])",
    )
    
    parser.add_argument(
        "--ilvr_x0",
        action='store_true',
        help="perform ILVR in x_0 space instead of x_t space",
    )
    
    parser.add_argument(
        "--mask_dilate",
        type=int,
        nargs='*',
        default=16,
        help="Dilate mask to contain larger region (# pixels to dilate)",
    )
    
    return parser

