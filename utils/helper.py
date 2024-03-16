from glob import glob
import cv2
import os


def load_sequence(directory):
    """
    Load all images from a sequence directory.

    :param directory: The path to the sequence directory.
    :return: A list of images (as NumPy arrays).
    """
    image_list = []

    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")

    image_files = glob(os.path.join(directory, '*.jpg')) + \
                  glob(os.path.join(directory, '*.png')) + \
                  glob(os.path.join(directory, '*.jpeg')) + \
                  glob(os.path.join(directory, '*.bmp'))

    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]

    for subdir in subdirectories:
        subdir_path = os.path.join(directory, subdir)
        image_files += glob(os.path.join(subdir_path, '*.jpg')) + \
                       glob(os.path.join(subdir_path, '*.png')) + \
                       glob(os.path.join(subdir_path, '*.jpeg')) + \
                       glob(os.path.join(subdir_path, '*.bmp'))

    if not image_files:
        raise ValueError(f"No image files found in the directory: {directory}")

    image_files.sort()  # Sort the image files to ensure the correct sequence

    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is not None:
            image_list.append(image)

    return image_list


def get_subfolders(directory):
    subfolders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    return subfolders

def check_distortion_exist(sequence_directory, distortion, severity, args):
    distortion_dir = os.path.join(args.output_dir, sequence_directory, distortion, str(severity))
    return os.path.exists(distortion_dir)

def count_jpg_files(directory):
    return len([file for file in os.listdir(directory) if file.endswith('.jpg')])

def check_distortion_num(sequence_directory, distortion, severity, args):
    distortion_dir = os.path.join(args.output_dir, sequence_directory, distortion, str(severity))
    seq_dir = os.path.join(args.seq_dir, sequence_directory)
    # Count the number of .jpg files in each directory
    count_distortion = count_jpg_files(distortion_dir)
    count_sequence = count_jpg_files(seq_dir)
    return count_distortion >= count_sequence