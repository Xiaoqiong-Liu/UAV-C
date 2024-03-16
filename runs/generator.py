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
from utils import *
from utils.helper import check_distortion_exist


def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return

def depth_estimation(args):
    # 1. get image lists
    subfolders = get_subfolders(args.seq_dir)
    subfolders.sort()
     # 2. load model checkpoint
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    # 3. move model to GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas = DataParallel(midas)  # Wrap the model with DataParallel
    midas.to(device)
    midas.eval()
    # 4. load transforms to resize and normalize the image for large or small model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    for sequence_directory in subfolders:
        images = load_sequence(f'{args.seq_dir}/{sequence_directory}')
        # 5. convert image to RGB and apply transforms
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        input_batch = [transform(img).to(device) for img in images]
        # 6. predict and resize to original resolution 
        # Loop through the input images
        for i, input_image in enumerate(input_batch):
            # print(images[0].shape)
            with torch.no_grad():
                prediction = midas(input_image)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=images[0].shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()
            output_filepath = os.path.join(args.output_dir, sequence_directory, 'depth_estimation')
            if not os.path.exists(output_filepath):
                    os.makedirs(output_filepath)
            output_filepath = f'{output_filepath}/depth_image_{i:06}'  # Specify the output file path
            # 7.save the prediction as grayscale
            write_depth(output_filepath, output, True, bits=1)
            print('writing image to ', output_filepath)

def generate(args):
    subfolders = get_subfolders(args.seq_dir)
    subfolders.sort()
    for i,sequence_directory in enumerate(subfolders):
        images = load_sequence(f'{args.seq_dir}/{sequence_directory}')
        distortions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        ]
        severity = 3
        depth_distortions = ['fog_shot_noise']
        MAP = {
            'fog': apply_fog_corruption,
            # 'fog_shot_noise': apply_fog_corruption_shot_noise
        }

        for distortion in distortions:
            frames = []
            idx = np.random.randint(5)
            for i in range(len(images)):
                frame = images[i]
                if distortion == 'frost':
                    corrupted_frame = corrupt(frame, severity, corruption_name=distortion, idx = idx)
                else:
                    corrupted_frame = corrupt(frame, severity, corruption_name=distortion)
                distortion_dir = os.path.join(args.output_dir, sequence_directory, distortion)
                if not os.path.exists(distortion_dir):
                    os.makedirs(distortion_dir)
                severity_dir = os.path.join(distortion_dir, str(severity))
                if not os.path.exists(severity_dir):
                    os.makedirs(severity_dir) 
                cv2.imwrite(f'{severity_dir}/image{i:06}_{distortion}.jpg', corrupted_frame)
                print('writing image to ', f'{severity_dir}/image{i:06}_{distortion}.jpg')
                # frames.append(corrupted_frame)

            # Save frames as a GIF
            # gif_dir = os.path.join(args.output_dir, sequence_directory, 'gif')
            # if not os.path.exists(gif_dir):
            #         os.makedirs(gif_dir)
            # output_filename = f'{gif_dir}/{distortion}.gif'
            # imageio.mimsave(output_filename, frames, duration=0.1)  # Adjust the duration as needed

        for distortion in depth_distortions:
            frames = []
            depth_maps = load_sequence(f'{args.output_dir}/{sequence_directory}/depth_estimation')
            for i in range(len(images)):
                frame = images[i]
                depth_map = depth_maps[i]
                depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
                corrupted_frame = MAP[distortion](frame, depth_map, severity)
                distortion_dir = os.path.join(args.output_dir, sequence_directory, distortion)
                if not os.path.exists(distortion_dir):
                    os.makedirs(distortion_dir)
                severity_dir = os.path.join(distortion_dir, str(severity))
                if not os.path.exists(severity_dir):
                    os.makedirs(severity_dir) 
                cv2.imwrite(f'{severity_dir}/image{i:06}_{distortion}.jpg', corrupted_frame)
                print('writing image to ', f'{severity_dir}/image{i:06}_{distortion}.jpg')
                # frames.append(corrupted_frame)

            # Save frames as a GIF
            # gif_dir = os.path.join(args.output_dir, sequence_directory, 'gif')
            # if not os.path.exists(gif_dir):
            #         os.makedirs(gif_dir)
            # output_filename = f'{gif_dir}/{distortion}.gif'
            # imageio.mimsave(output_filename, frames, duration=0.1)  # Adjust the duration as needed

def generate2(args):
    subfolders = get_subfolders(args.seq_dir)
    subfolders.sort()
    for i,sequence_directory in enumerate(subfolders):
        images = load_sequence(f'{args.seq_dir}/{sequence_directory}')
        # distortions = [
        #     'gaussian_noise', 'shot_noise', 'impulse_noise',
        #     'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        #     'snow', 'frost', 'fog', 'brightness',
        #     'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        #     'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        # ]
        distortions = [
            # 'snow', 'frost', 'brightness',
            # 'contrast', 'elastic_transform', 'pixelate'
        ]
        severity = 5
        depth_distortions = ['fog_speckle_noise']
        MAP = {
            'fog_speckle_noise': fog_speckle_noise
        }

        for distortion in distortions:
            frames = []
            idx = np.random.randint(5)
            for i in range(len(images)):
                frame = images[i]
                if distortion == 'frost':
                    corrupted_frame = corrupt(frame, severity, corruption_name=distortion, idx = idx)
                else:
                    corrupted_frame = corrupt(frame, severity, corruption_name=distortion)
                distortion_dir = os.path.join(args.output_dir, sequence_directory, distortion)
                if not os.path.exists(distortion_dir):
                    os.makedirs(distortion_dir)
                severity_dir = os.path.join(distortion_dir, str(severity))
                if not os.path.exists(severity_dir):
                    os.makedirs(severity_dir) 
                cv2.imwrite(f'{severity_dir}/image{i:06}_{distortion}.jpg', corrupted_frame)
                print('writing image to ', f'{severity_dir}/image{i:06}_{distortion}.jpg')
                # frames.append(corrupted_frame)
        for distortion in depth_distortions:
            frames = []
            depth_maps = load_sequence(f'{args.output_dir}/{sequence_directory}/depth_estimation')
            for i in range(len(images)):
                frame = images[i]
                depth_map = depth_maps[i]
                depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
                print(frame.shape, depth_map.shape)
                corrupted_frame = MAP[distortion](frame, depth_map, severity)
                distortion_dir = os.path.join(args.output_dir, sequence_directory, distortion)
                if not os.path.exists(distortion_dir):
                    os.makedirs(distortion_dir)
                severity_dir = os.path.join(distortion_dir, str(severity))
                if not os.path.exists(severity_dir):
                    os.makedirs(severity_dir) 
                cv2.imwrite(f'{severity_dir}/image{i:06}_{distortion}.jpg', corrupted_frame)
            # Save frames as a GIF
            # gif_dir = os.path.join(args.output_dir, sequence_directory, 'gif')
            # if not os.path.exists(gif_dir):
            #         os.makedirs(gif_dir)
            # output_filename = f'{gif_dir}/{distortion}.gif'
            # imageio.mimsave(output_filename, frames, duration=0.1)  # Adjust the duration as needed   

def generate3(args):
    subfolders = get_subfolders(args.seq_dir)
    subfolders.sort()
    # parrallel the corruptions
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = []
        
        for sequence_directory in subfolders:
            print(sequence_directory)
            images = load_sequence(f'{args.seq_dir}/{sequence_directory}')

            distortions = [
                'clean']
            severitys = [3]
            for distortion in distortions:
                for severity in severitys:
                    if(not check_distortion_exist(sequence_directory, distortion, severity, args)):
                        futures.append(executor.submit(distort, args, sequence_directory, distortion, images, severity))
            # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    

def generate4(args):
    subfolders = get_subfolders(args.seq_dir)
    subfolders.sort()

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = []
        
        for sequence_directory in subfolders:
            print(sequence_directory)
            images = load_sequence(f'{args.seq_dir}/{sequence_directory}')
            
            distortions = [
                # 'gaussian_noise', 'shot_noise', 'impulse_noise',
                # 'defocus_blur','motion_blur',
                # 'frost', 'snow', 'fog','rain'
                "zoom_blur"
                # 'contrast', 
                # 'speckle_noise', 'gaussian_blur', 'spatter'
            ]
            severitys = [1,3,5]
            for distortion in distortions:
                for severity in severitys: 
                    futures.append(executor.submit(distort, args, sequence_directory, distortion, images, severity))
                    # if(not check_distortion_exist(sequence_directory, distortion, severity, args)):
                        # futures.append(executor.submit(distort, args, sequence_directory, distortion, images, severity))
                    # elif(not check_distortion_num(sequence_directory, distortion, severity, args)):
                    # if(sequence_directory == 'wakeboard6'):
                        # futures.append(executor.submit(distort, args, sequence_directory, distortion, images, severity))
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)



def generate5(args):
    subfolders = get_subfolders(args.seq_dir)
    subfolders.sort()

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = []
        
        for sequence_directory in subfolders:
            print(sequence_directory)
            images = load_sequence(f'{args.seq_dir}/{sequence_directory}')
            
            distortions = [
                'rain_defocusBlur', 'rain_gaussianNoise', 'defocusBlur_gaussianNoise', 'rain_defocusBlur_gaussianNoise'
            ]
            severitys = [1,3,5]
            for distortion in distortions:
                for severity in severitys:
                    if(not check_distortion_exist(sequence_directory, distortion, severity, args)):
                        futures.append(executor.submit(distort, args, sequence_directory, distortion, images, severity))
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)


