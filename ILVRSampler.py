
from omegaconf import OmegaConf, ListConfig
import torch
from torch import autocast
from tqdm import tqdm, trange
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation
from pytorch_lightning import seed_everything
from einops import rearrange

from CS2Real.ldm.util import instantiate_from_config
from CS2Real.ldm.models.diffusion.ddim import DDIMSampler
from CS2Real.dev.prompt_masking import masked_cross_attention



class ILVRSampler():
    def __init__(self, config) -> None:
        self.down_N_in = config.down_N_in
        self.down_N_out = config.down_N_out
        self.T_in = config.T_in
        self.T_out = config.T_out

        sd_config_path = "/disk2/nadav/source/CS2Real/configs/stable-diffusion/v1-inference.yaml"
        ckpt_path = "/disk2/nadav/source/CS2Real/models/ldm/stable-diffusion-v1/model.ckpt"
        sd_config = OmegaConf.load(f"{sd_config_path}")

        self.model = self._load_model_from_config(sd_config, f"{ckpt_path}")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.sampler = DDIMSampler(self.model)

        self.precision_scope = autocast 
        self.seed = config.seed
        self.scale = config.scale
        self.ddim_steps = config.ddim_steps
        self.dilate = config.mask_dilate
        self.image_size = (config.image_size, config.image_size)
        self.n_samples = config.n_samples
        self.blend_pix = config.blend_pix

        # self.prompt_in = None
        # self.prompt_out = None
        # self.prompt_amplifier_in = 1.
        # self.prompt_amplifier_out = 1.
        self.strength_in = 1.0
        self.strength_out = None
        self.pixel_cond_space = False
        self.ilvr_x0 = False
        self.ddim_eta = 0.0


        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)
        self.repaint_conf = OmegaConf.create({'use_repaint': config.repaint_start > 0,
                                                'inpa_inj_time_shift': 1,
                                                'schedule_jump_params': {
                                                't_T': self.ddim_steps,
                                                'n_sample': 1,
                                                'jump_length': 10,
                                                'jump_n_sample': 10,
                                                'start_resampling': int(config.repaint_start * self.ddim_steps),
                                                'collapse_increasing': False}})


    def _load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        return model



    def load_img(self, image, size, crop_bbox=None):
        # w, h = image.size
        # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32.
        if crop_bbox is not None:
            image = image.crop(crop_bbox)
            
        if image.size != size:
            image = image.resize(size, resample=Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2. * image - 1.


    def get_mask_bbox(self, mask):
        mask = np.array(mask)

        indices = np.where(mask>=128)
        min_x = indices[1].min()
        max_x = indices[1].max()
        min_y = indices[0].min()
        max_y = indices[0].max()

        return (min_x, min_y, max_x, max_y)

   
    def load_mask(self,mask, size, dilate, crop_bbox=None):
        # mask = Image.open(path).convert("L")
        mask = mask.convert("L")
        if crop_bbox is not None:
            mask = mask.crop(crop_bbox)

        if mask.size != size:
            mask = mask.resize(size, resample=Image.Resampling.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
    
        mask = mask[None, None]
        mask[mask < 0.05] = 0
        mask[mask >= 0.05] = 1
        if dilate > 0:
            mask = binary_dilation(mask, iterations=dilate).astype(np.float32)
        mask = 1 - mask
        mask = torch.from_numpy(mask)
        return mask


    # def crop_small_object(self, img, mask):
    #     pass


    def load_data(self, img_list, mask_list, prompt_list):
        if not isinstance(img_list, list):
            img_list = [img_list]
        if not isinstance(mask_list, list):
            mask_list = [mask_list]
        if not isinstance(prompt_list, list):
            prompt_list = [prompt_list]

        assert len(img_list) == len(mask_list) 
        n_images = len(img_list)
        if len(prompt_list) < n_images:
            prompt_list = [""] * n_images

        torch_img_list = []
        torch_mask_list = []
        img_orig_size_list = []
        main_img_list = []
        for img, mask in zip(img_list, mask_list):
            min_bbox_x, min_bbox_y, max_bbox_x, max_bbox_y = self.get_mask_bbox(mask)
            bbox_size = (max_bbox_x-min_bbox_x)*(max_bbox_y-min_bbox_y)
            img_size = img.size[0]*img.size[1]

            crop_bbox = None
            if bbox_size / img_size <= 0.1:
                bbox_x_length = max_bbox_x-min_bbox_x
                bbox_y_length = max_bbox_y-min_bbox_y
                crop_bbox = (
                    max(0, min_bbox_x-bbox_x_length),
                    max(0, min_bbox_y-bbox_y_length),
                    min(img.size[0], max_bbox_x+bbox_x_length),
                    min(img.size[1], max_bbox_y+bbox_y_length),
                    )
                img_orig_size_list.append((crop_bbox[2]-crop_bbox[0], crop_bbox[3]-crop_bbox[1]))
                torch_img_list.append(self.load_img(img, size=self.image_size,crop_bbox=crop_bbox))
                torch_mask_list.append(self.load_mask(mask, size=self.image_size ,dilate=self.dilate,crop_bbox=crop_bbox))
                main_img_list.append((np.array(img).astype(np.float32), crop_bbox))
            else:
                img_orig_size_list.append(img.size)
                torch_img_list.append(self.load_img(img, size=self.image_size))
                torch_mask_list.append(self.load_mask(mask, size=self.image_size ,dilate=self.dilate))
                main_img_list.append(None)


        return torch_img_list, torch_mask_list, prompt_list, img_orig_size_list, main_img_list

    def sample(self, img_list, mask_list, prompt_list):
        curr_img_list, curr_mask_list, curr_prompt_list, img_orig_size_list, orig_img_list = self.load_data(img_list, mask_list, prompt_list)

        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    # for img, mask, prompts in tqdm(data_loader, desc="Data"):
                    for img, mask, prompts, orig_size, orig_img_n_bbox in tqdm(zip(curr_img_list, curr_mask_list, curr_prompt_list, img_orig_size_list, orig_img_list)):
                        if self.seed:
                            seed_everything(self.seed)
                        img, mask = img.to(self.device), mask.to(self.device)
                        batch_size = img.shape[0]
                        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(img))  # move to latent space
                        uc = None
                        if self.scale != 1.0:
                            uc = self.model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.model.get_learned_conditioning(prompts)

                        tokenizer = self.model.cond_stage_model.tokenizer
                        latent_mask = torch.nn.functional.interpolate(mask, size=init_latent.shape[-2:]).to(self.device) if mask is not None else None
                        # masked_cross_attention(self.model, prompts, self.prompt_in, self.prompt_out,
                        #                        latent_mask, tokenizer, self.prompt_amplifier_in,
                        #                        self.prompt_amplifier_out)

                        for n in trange(self.n_samples, desc="Sampling"):
                            # encode (scaled latent)
                            t_enc_in = int(self.strength_in * self.ddim_steps)
                            if t_enc_in < self.ddim_steps:
                                z_enc_in = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc_in]*batch_size).to(device))
                            else:  # strength >= 1 ==> use only noise
                                z_enc_in = torch.randn_like(init_latent)
                            t_enc_out = int(self.strength_out * self.ddim_steps) if self.strength_out is not None else t_enc_in
                            if t_enc_out < self.ddim_steps:
                                z_enc_out = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc_out] * batch_size).to(device))
                            else:  # strength >= 1 ==> use only noise
                                z_enc_out = torch.randn_like(init_latent)
                            z_enc = latent_mask * z_enc_out + (1 - latent_mask) * z_enc_in if latent_mask is not None else z_enc_out


                            # decode it
                            samples = self.sampler.decode(z_enc, c, t_enc_in,
                                                    unconditional_guidance_scale=self.scale,
                                                    unconditional_conditioning=uc, ref_image=img, mask=mask,
                                                    t_start_out=t_enc_out, down_N_out=self.down_N_out,
                                                    down_N_in=self.down_N_in, T_out=self.T_out,
                                                    T_in=self.T_in, blend_pix=self.blend_pix,
                                                    pixel_cond_space=self.pixel_cond_space,
                                                    repaint=self.repaint_conf, ilvr_x0=self.ilvr_x0)

                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            x_samples = 255. * rearrange(x_samples.cpu().numpy()[0], 'c h w -> h w c')
                            x_samples = Image.fromarray(x_samples.astype(np.uint8))
                            if x_samples.size != orig_size:
                                x_samples = x_samples.resize(orig_size, resample=Image.Resampling.NEAREST)
                            
                            
                            if orig_img_n_bbox is not None:
                                repasted_img = orig_img_n_bbox[0]
                                bbox = orig_img_n_bbox[1]

                                repasted_img[bbox[1]:bbox[3],bbox[0]:bbox[2],:] = np.array(x_samples).astype(np.float32)
                                x_samples = repasted_img
                                x_samples = Image.fromarray(x_samples.astype(np.uint8))


                            all_samples.append(x_samples)
                       
                    return all_samples