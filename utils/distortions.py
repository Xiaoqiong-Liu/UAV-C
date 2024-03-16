# from imagenet_c import corrupt 
import os
import cv2
from imagenet_c import corrupt 
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import concurrent.futures
from glob import glob
from torch.nn.parallel import DataParallel
import imgaug.augmenters as iaa
from scipy.ndimage import zoom as scizoom
import random

def rain(image, severity, seed=None):
    # Determine density based on severity
    density = [
        (0.01, 0.06),
        (0.06, 0.10),
        (0.10, 0.15),
        (0.15, 0.20),
        (0.20, 0.25),
    ][severity - 1]

    # Configure rain augmentation
    iaa_seq = iaa.Sequential([
        iaa.RainLayer(
            density=density,
            density_uniformity=(0.8, 1.0),
            drop_size=(0.4, 0.6),
            drop_size_uniformity=(0.10, 0.20),
            angle=(-15, 15),
            speed=(0.04, 0.20),
            blur_sigma_fraction=(0.0001, 0.001),
            blur_sigma_limits=(0.5, 3.75),
            seed=seed
        )
    ])

    image_np = np.array(image)

    # Add rain
    images = image_np[None]
    images_aug = iaa_seq(images=images)
    image_aug = images_aug[0]

    # Simulate rainy appearance
    gray_ratio = 0.3
    image_aug = gray_ratio * np.ones_like(image_aug) * 128 + (1 - gray_ratio) * image_aug
    image_aug = image_aug.astype(np.uint8)

    # Lower brightness
    image_rgb_255 = image_aug
    img_hsv = cv2.cvtColor(image_rgb_255, cv2.COLOR_RGB2HSV).astype(np.int64)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] / img_hsv[:, :, 2].max() * 256 * 0.7
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)
    img_hsv = img_hsv.astype(np.uint8)
    image_rgb_255 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    image_aug = image_rgb_255

    return image_aug

def apply_fog_corruption(input_image, depth_map, severity = 3, fog_color=None, absorption_coeff=0.015, scattering_coeff=0.008):
    """
    Apply fog corruption to an input image and its corresponding depth map.

    Args:
        input_image (numpy.ndarray): The input image as a NumPy array (H x W x C).
        depth_map (numpy.ndarray): The depth map as a NumPy array (H x W).
        fog_density (float): Controls the density of the fog effect. A higher value results in thicker fog. Default is 0.2.
        fog_color (numpy.ndarray): The color of the fog in RGB format. Default is None, which means fog color is white [255, 255, 255].
        absorption_coeff (float): Controls the rate of absorption. Default is 0.1.
        scattering_coeff (float): Controls the rate of scattering. Default is 0.05.

    Returns:
        numpy.ndarray: The corrupted image with fog applied.
    """
    fog_density = severity*0.1
    # Ensure input_image and depth_map have the same dimensions
    assert input_image.shape[:2] == depth_map.shape, "Input image and depth map must have the same dimensions."

    # Calculate optical depth based on depth and coefficients
    optical_depth = (255+100-depth_map) * (absorption_coeff+scattering_coeff)

    # Calculate transmission based on fog density and optical depth
    transmission = np.exp(-fog_density * optical_depth)

    # Simulate fog color based on fog density
    if fog_color is None:
        fog_color = np.array([225, 225, 201])  # Default fog color is gray
    else:
        assert fog_color.shape == (3,), "Fog color must be a 1D array of length 3."

    corrupted_image = input_image * transmission[:, :, np.newaxis] + fog_color * (1 - transmission[:, :, np.newaxis])

    corrupted_image = np.clip(corrupted_image, 0, 255).astype(np.uint8)

    return corrupted_image

def apply_fog_corruption_shot_noise(input_image, depth_map, severity = 3):
    image = apply_fog_corruption(input_image, depth_map, severity)
    image = corrupt(image, severity, corruption_name='shot_noise')
    return image

def fog_speckle_noise(input_image, depth_map, severity = 3):
    image = apply_fog_corruption(input_image, depth_map, severity)
    image = corrupt(image, severity, corruption_name='speckle_noise')
    return image

def clipped_zoom_blur_seq(images, severity=1, i=0):
    stride = [0.03,0.02,0.01][i % 3]
    c = [np.arange(1, 1.05+[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.6,0.5,0.4,0.3,0.2,0.1][i%13], stride),
         np.arange(1, 1.16, stride),
         np.arange(1, 1.15+[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.6,0.5,0.4,0.3,0.2,0.1][i%13], stride),
         np.arange(1, 1.26, stride),
         np.arange(1, 1.25+[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.6,0.5,0.4,0.3,0.2,0.1][i%13], stride)][severity - 1]
    x = images[i]
    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    lgth = 0
    for index,zoom_factor in enumerate(c):
        j = min(i,1)
        y = images[i-j]
        y = np.array(y / 255.).astype(np.float32)    
        out += clipped_zoom(y, zoom_factor)
        lgth = lgth+1
    x = (x + out) / (lgth + 1)
    x =np.clip(x, 0, 1) * 255
    return x

def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    w = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))

    top_h = (h - ch) // 2
    top_w = (w - cw) // 2
    img = scizoom(img[top_h:top_h + ch, top_w:top_w + cw], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top_h = (img.shape[0] - h) // 2
    trim_top_w = (img.shape[1] - w) // 2

    return img[trim_top_h:trim_top_h + h, trim_top_w:trim_top_w + w]

def rain_defoucsBlur(image, severity):
    image = rain(image, severity)
    image = corrupt(image, severity, corruption_name = 'defocus_blur')
    return image

def rain_gasussianNoise(image, severity):
    image = rain(image, severity)
    image = corrupt(image, severity, corruption_name = 'gaussian_noise')
    return image

def defocusBlur_gaussianNoise(image, severity):
    image = corrupt(image, severity, corruption_name = 'defocus_blur')
    image = corrupt(image, severity, corruption_name = 'gaussian_noise')
    return image

def rain_defoucsBlur_gaussianNoise(image, severity):
    image = rain(image, severity)
    image = corrupt(image, severity, corruption_name = 'defocus_blur')
    image = corrupt(image, severity, corruption_name = 'gaussian_noise')
    return image
                 