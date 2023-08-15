import argparse
from dataclasses import field, dataclass
from typing import Callable, Optional, Any, Dict
from skimage import io
import os
import pathlib
import numpy as np
from inpaint_config import InPaintConfig
from inpaint_tools import read_file_list
from tqdm import tqdm
from diffusers import (
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler
)
from diffusers.utils import load_image
import torch
from PIL import Image

@dataclass
class Directories:
    input_data_dir: str
    output_data_dir: str
    dataset_split: str
    model_dir: str = field(init=False)
    inpainted_result_dir: str = field(init=False)

    def __post_init__(self):
        self.model_dir = os.path.join(self.output_data_dir, "trained_model")
        self.inpainted_result_dir = os.path.join(
            self.output_data_dir, f"inpainted_{self.dataset_split}"
        )
        pathlib.Path(self.inpainted_result_dir).mkdir(parents=True, exist_ok=True)


def inpaint_one_image(in_image, mask_image, avg_image):
    mask_image = np.squeeze(mask_image)
    inpainted_mask = np.copy(avg_image)
    inpainted_mask[mask_image == 0] = 0

    inpaint_image = inpainted_mask + in_image
    return inpaint_image



def inpaint_with_diffusers(prompt: str, dirs: Directories, file_ids: list[str]):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to("cuda")
    
    for idx in tqdm(file_ids):
        in_image_name = os.path.join(
            dirs.input_data_dir, "masked", f"{idx}_stroke_masked.png"
        )
        in_mask_name = os.path.join(dirs.input_data_dir, "masks", f"{idx}_stroke_mask.png")
        out_image_name = os.path.join(dirs.inpainted_result_dir, f"{idx}.png")

        im_masked = Image.open(in_image_name)
        im_mask = Image.open(in_mask_name)

 
        inpainted_image = pipe(prompt=prompt, image=im_masked, mask_image=im_mask, num_inference_steps).images[0]
        inpainted_image.save(out_image_name)




def setup_directories(settings) -> Directories:
    dirs = Directories(
        input_data_dir=settings["dirs"]["input_data_dir"],
        output_data_dir=settings["dirs"]["output_data_dir"],
        dataset_split=settings["data_set"],
    )
    return dirs

def get_file_list(dirs: Directories) -> list[str]:
    file_list = os.path.join(dirs.input_data_dir, "data_splits", dirs.dataset_split + ".txt")
    file_ids = read_file_list(file_list)
    return file_ids

def inpaint_images(dirs: Directories, inpaint_loop_fn: Callable, inpaint_loop_fn_kwargs: Dict[str, Any]):
    print(
        f"InPainting {dirs.dataset_split} and placing results in {dirs.inpainted_result_dir} with model from {dirs.model_dir}"
    )

    file_ids = get_file_list(dirs=dirs)
    print(f"Inpainting {len(file_ids)} images")
    inpaint_loop_fn(dirs=dirs, file_ids=file_ids, **inpaint_loop_fn_kwargs)




if __name__ == "__main__":
    args = argparse.ArgumentParser(description="InpaintImages")
    config = InPaintConfig(args)
    if config.settings is not None:
        dirs = setup_directories(config.settings)
        prompt = "The face of a cat"
        inpaint_images(dirs=dirs, inpaint_loop_fn=inpaint_with_diffusers, inpaint_loop_fn_kwargs={"prompt" : prompt})
