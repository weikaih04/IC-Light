# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import json

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file
from tqdm import tqdm
import json

from gradio_demo import (
    hooked_unet_forward, 
    encode_prompt_pair, 
    pytorch2numpy, 
    numpy2pytorch, 
    resize_and_center_crop, 
    resize_without_crop, 
)

def parse_rgba(img, sigma=0.0):
    # Load the RGBA image
    
    # Ensure the input image is RGBA
    assert img.shape[2] == 4, "Input image must have 4 channels (RGBA)."
    
    # Separate the RGB and alpha channels
    rgb = img[:, :, :3]  # Extract RGB channels
    alpha = img[:, :, 3]  # Extract alpha channel
    
    # Normalize the alpha channel to [0, 1]
    alpha = alpha.astype(np.float32) / 255.0  # Alpha is either 0 (transparent) or 1 (opaque)

    # Blend the RGB image using the alpha mask
    result = 127 + (rgb.astype(np.float32) - 127 + sigma) * alpha[:, :, None]  # Apply alpha blending

    # Clip the result to valid range and return
    return result.clip(0, 255).astype(np.uint8), alpha


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward

unet.forward = hooked_unet_forward

# Load

model_path = './models/iclight_sd15_fc.safetensors'

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys
# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    bg_source = BGSource(bg_source)
    input_bg = None

    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong initial latent!'

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    if input_bg is None:
        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample

    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    input_fg, matting = parse_rgba(input_fg)
    results = process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
    return input_fg, results




def adjust_dimensions(width, height, max_dim=1024, divisible_by=8):
    """
    Adjust width and height to maintain the original aspect ratio, 
    cap at max_dim, and make them divisible by a specified value.
    """
    # Calculate aspect ratio
    # aspect_ratio = width / height

    # # Determine scaling factor to cap at max_dim
    # if width > height:
    #     scaled_width = min(width, max_dim)
    #     scaled_height = scaled_width / aspect_ratio
    # else:
    #     scaled_height = min(height, max_dim)
    #     scaled_width = scaled_height * aspect_ratio

    # # Ensure divisibility by the specified value
    # scaled_width = int((scaled_width // divisible_by) * divisible_by)
    # scaled_height = int((scaled_height // divisible_by) * divisible_by)

    # return scaled_width, scaled_height
    return 1024, 1024


def main(args):

    data_path = args.dataset_path
    output_data_path = args.output_data_path
    illuminate_prompts_path = args.illuminate_prompts_path
    illuminate_prompts = json.load(open(illuminate_prompts_path))
    
    record_path = args.record_path
    records = {}

    # Ensure the output directory exists
    os.makedirs(output_data_path, exist_ok=True)

    if args.index_json_path is not None:
        # Load the list of filenames from the JSON file
        with open(args.index_json_path, 'r') as f:
            all_filenames = json.load(f)
        
        # Validate that all_filenames is a list
        if not isinstance(all_filenames, list):
            raise ValueError("The index JSON file must contain a list of filenames.")
        
        if args.num_splits < 1:
            raise ValueError("num_splits must be at least 1.")
        if args.split < 0 or args.split >= args.num_splits:
            raise ValueError(f"split index must be between 0 and {args.num_splits - 1}.")

        # Split the list into num_splits parts
        splits = np.array_split(all_filenames, args.num_splits)
        split_filenames = splits[args.split]

        print(f"Processing split {args.split + 1}/{args.num_splits} with {len(split_filenames)} images.")
    else:
        # If no index_json_path is provided, process all .png files in the dataset_path
        split_filenames = [fg_name for fg_name in os.listdir(data_path) if fg_name.endswith(".png")]
        print(f"Processing all {len(split_filenames)} images in the dataset.")

    for fg_name in tqdm(split_filenames):
        input_fg_path = os.path.join(data_path, fg_name)            
        output_path = os.path.join(output_data_path, f"{os.path.splitext(fg_name)[0]}.jpg")
        if os.path.exists(output_path):
            # print(f"Skipping '{fg_name}': Output file '{output_path}' already exists.")
            continue
        # Load input images
        input_fg = np.array(Image.open(input_fg_path))

        # Dynamically calculate dimensions
        image_height, image_width = input_fg.shape[:2]

        # Adjust dimension to maintain aspect ratio and make them divisible by 8
        image_width, image_height = adjust_dimensions(image_width, image_height, max_dim=1024, divisible_by=8)

        print(f"Processing '{fg_name}': Adjusted dimensions: {image_width} x {image_height}")

        # Define parameters

        # randomly sample from prompts_list
        prompt = np.random.choice(illuminate_prompts)
        bg_source = np.random.choice([BGSource.NONE, BGSource.NONE, BGSource.NONE, BGSource.NONE, BGSource.LEFT, BGSource.RIGHT, BGSource.TOP, BGSource.BOTTOM])
        # record the choice for each image
        

        seed = 12345
        steps = 25
        a_prompt = "not obvious objects in the background, best quality"
        n_prompt = "have obvious objects in the background, lowres, bad anatomy, bad hands, cropped, worst quality"
        cfg = 2.0
        highres_scale = 1.0
        highres_denoise = 0.5
        lowres_denoise = 0.9
        num_samples = 1  # Adjust as needed

        # Process and save the result
        _, results = process_relight(
            input_fg=input_fg,
            prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            cfg=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
            lowres_denoise=lowres_denoise, 
            bg_source=bg_source
        )

        # Save the output image
        Image.fromarray(results[0]).save(output_path)
        print(f"Saved relit image to '{output_path}'")
        
        records[fg_name] = {
            "output_path": output_path,
            "prompt": prompt,
            "bg_source": bg_source.value,  # Store the Enum as string
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "highres_scale": highres_scale,
            "highres_denoise": highres_denoise,
            "lowres_denoise": lowres_denoise
        }
        

    # if os.path.exists(record_path):
    #     with open(record_path, 'r') as f:
    #         all_records = json.load(f)
    # else:
    #     all_records = {}
    # all_records.update(records)
    # with open(record_path, 'w') as f:
    #     json.dump(all_records, f, indent=4)




if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description="Relight images using Stable Diffusion pipelines.")
        parser.add_argument('--dataset_path', type=str, required=True, help="Path to the segment dataset")
        parser.add_argument('--output_data_path', type=str, required=True, help="Path to the output data")
        parser.add_argument('--num_splits', type=int, default=1, help="Number of splits to create")
        parser.add_argument('--split', type=int, default=0, help="Split index to process")
        parser.add_argument('--index_json_path', type=str, default=None, help="Path to the JSON file containing the image filenames")
        parser.add_argument('--illuminate_prompts_path', type=str, default=None, help="Path to the JSON file containing the illumination prompts")
        parser.add_argument('--record_path', type=str, default=None, help="Path to the JSON file containing the record of the prompts used")
        return parser.parse_args()
    
    args = parse_args()
    main(args)