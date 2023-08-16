import argparse
from skimage import io
import os
import pathlib
import numpy as np
from inpaint_config import InPaintConfig
from inpaint_tools import read_file_list
from tqdm import tqdm

import matplotlib.pyplot as plt

def knn(img, k, mask, train_imgs, input_data_dir):
    assert k > 0

    k_best_dist = np.inf
    k_best_files = [(None, np.inf)]*k

    for idx, train_img in train_imgs:
        train_img[mask == 1] = 0
        dist = np.linalg.norm(img-train_img)

        if dist < k_best_dist:
            k_best_files.append((idx, dist))
            k_best_files.sort(key=lambda el: el[1])
            k_best_files = k_best_files[:k]
            k_best_dist = k_best_files[k-1][1]

            assert len(k_best_files) == k

    # calculate average image
    avg_img = np.zeros(img.shape)

    for k_img_idx, dist in k_best_files:
        k_img_path = os.path.join(input_data_dir, "originals", f"{k_img_idx}.jpg")
        k_img = io.imread(k_img_path)
        avg_img += k_img

    avg_img /= k

    return avg_img.astype(np.uint8)


def load_images(input_data_dir, data_set="training"):
    file_list = os.path.join(input_data_dir, "data_splits", data_set + ".txt")
    file_ids = read_file_list(file_list)
    files = []

    for idx in tqdm(file_ids):
        train_img_path = os.path.join(input_data_dir, "originals", f"{idx}.jpg")
        train_img = io.imread(train_img_path)
        files.append((idx, train_img))

    return files

def inpaint_one_image(in_image, k, mask, train_imgs, input_data_dir):
    mask = np.squeeze(mask)
    inpainted_mask = knn(in_image, k, mask, train_imgs, input_data_dir)
    inpainted_mask[mask == 0] = 0

    inpaint_image = inpainted_mask + in_image
    return inpaint_image

def inpaint_images(settings):
    input_data_dir = settings["dirs"]["input_data_dir"]
    output_data_dir = settings["dirs"]["output_data_dir"]
    data_set = settings["data_set"]
    model_dir = os.path.join(output_data_dir, "trained_model")
    k = settings["training_parms"]["k"]

    print(f"Inpainting with k={k}")

    inpainted_result_dir = os.path.join(output_data_dir, f"inpainted_{data_set}")
    pathlib.Path(inpainted_result_dir).mkdir(parents=True, exist_ok=True)

    print(f"InPainting {data_set} and placing results in {inpainted_result_dir} with model from {model_dir}")

    file_list = os.path.join(input_data_dir, "data_splits", data_set + ".txt")
    file_ids = read_file_list(file_list)
    if file_ids is None:
        return

    print("Loading training files")
    train_imgs = load_images(input_data_dir)

    print(f"Inpainting {len(file_ids)} images")

    for idx in tqdm(file_ids):
        in_image_name = os.path.join(input_data_dir, "masked", f"{idx}_stroke_masked.png")
        in_mask_name = os.path.join(input_data_dir, "masks", f"{idx}_stroke_mask.png")
        out_image_name = os.path.join(inpainted_result_dir, f"{idx}.png")

        im_masked = io.imread(in_image_name)
        im_mask = io.imread(in_mask_name)

        inpainted_image = inpaint_one_image(in_image=im_masked, mask=im_mask, k=k, train_imgs=train_imgs, input_data_dir=input_data_dir)
        io.imsave(out_image_name, inpainted_image)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='InpaintImages')
    config = InPaintConfig(args)
    if config.settings is not None:
        inpaint_images(config.settings)