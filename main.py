import numpy as np
from PIL import Image
import cv2
import warnings
import os
import argparse
import matplotlib.pyplot as plt
# from imagenet_c import corrupt 


'''
Add XX noise
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--output_dir', type=str, default='/mnt/Data/XLDataset/UAV123_10fps-C/data_corrupt', help='the directory to write corruption results')
    parser.add_argument('--seq_dir', type=str, default='/mnt/Data/XLDataset/DTB70', help='the directory to write corruption results')
    # runs
    parser.add_argument('--phase', type=str, default='generate', choices=['generate','depth_estimation'])
    args = parser.parse_args()
    

    from runs.generator import *
    if args.phase == 'generate':
        generate(args)
    if args.phase == 'depth_estimation':
        depth_estimation(args)
    

