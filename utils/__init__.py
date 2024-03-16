import numpy as np
from .distortions import *
from imagenet_c import corrupt
from .helper import *

Distortions1 = (
    'clean','gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'rain',
    'contrast', 
    'speckle_noise', 'gaussian_blur', 'spatter'
)
Depth_distortions =(
    'fog', 'fog_defocusBlur', 'fog_defocusBlur_gaussianNoise'
)
Composite_distortions = (
    'rain_defocusBlur', 'rain_gaussianNoise', 'defocusBlur_gaussianNoise', 
    'rain_defocusBlur_gaussianNoise'
)

def generate_composite_distortions(args, sequence_directory, distortion, images, severity):
    # print('generate composite distributions',distortion)
    distortion_path = f'{args.output_dir}/{sequence_directory}/{distortion}'
    # create distortion path
    if not os.path.exists(distortion_path):
        os.makedirs(distortion_path)
    severity_dir = os.path.join(distortion_path, str(severity))
    if not os.path.exists(severity_dir):
        os.makedirs(severity_dir) 

    MAP = {
            'rain_defocusBlur': rain_defoucsBlur,
            'rain_gaussianNoise': rain_gasussianNoise,
            'rain_defocusBlur_gaussianNoise': rain_defoucsBlur_gaussianNoise,
            'defocusBlur_gaussianNoise': defocusBlur_gaussianNoise
        }
    # step 2: generate frames for distortion
    for i in range(len(images)):
        frame = images[i]
        corrupted_frame = MAP[distortion](frame, severity)
        cv2.imwrite(f'{severity_dir}/image{i:06}_{distortion}.jpg', corrupted_frame)
        print('writing image to ', f'{severity_dir}/image{i:06}_{distortion}.jpg')


def generate_depth_distortion(args, sequence_directory, distortion, images, severity):
    MAP = {
            'fog': apply_fog_corruption,
            'fog_shot_noise': apply_fog_corruption_shot_noise
        }
    depth_maps_path = f'{args.output_dir}/{sequence_directory}/depth_estimation'

    # Assert that the depth path does not exist
    assert os.path.exists(depth_maps_path), f"Path does not exist: {depth_maps_path}"
    depth_maps = load_sequence(depth_maps_path)

    # Assert that the images list is not empty
    assert len(depth_maps) > 0, f"No images found in: {depth_maps_path}"
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

def generate_distortion(args, sequence_directory, distortion, images, severity):
    frames = []
    idx = np.random.randint(5)
    distortion_dir = os.path.join(args.output_dir, sequence_directory, distortion)
        
    # save distortion
    if not os.path.exists(distortion_dir):
        os.makedirs(distortion_dir)
    
    severity_dir = os.path.join(distortion_dir, str(severity))
    if not os.path.exists(severity_dir):
        os.makedirs(severity_dir) 

    
    for i in range(len(images)):
        print(i)
        frame = images[i]
        # generate frames
        if distortion == 'frost':
            corrupted_frame = corrupt(frame, severity, corruption_name=distortion, idx=idx)
        elif distortion == 'clean':
            corrupted_frame = frame
        elif distortion == 'rain':
            corrupted_frame = rain(frame, severity)
        elif distortion == 'zoom_blur':
            corrupted_frame = clipped_zoom_blur_seq(images, severity, i)
        else:
            corrupted_frame = corrupt(frame, severity, corruption_name=distortion)
        
        cv2.imwrite(f'{severity_dir}/image{i:06}_{distortion}.jpg', corrupted_frame)
        print('writing image to ', f'{severity_dir}/image{i:06}_{distortion}.jpg')

# Map imageNet-C distortions to the appropriate function (corrupt)
corruption_dict = {distortion: generate_distortion for distortion in Distortions1}
corruption_dict.update({distortion: generate_composite_distortions for distortion in Composite_distortions})
# Map depth distortions to the appropriate function (distort_depth_image)

corruption_dict.update({depth_distortion: generate_depth_distortion for depth_distortion in Depth_distortions})

def distort(args, sequence_directory, corruption_name, images, severity): # a wrapper for the distortions
    """
    :param severity: strength with which to corrupt x; an integer in (1,3,5)
    """

    corruption_dict[corruption_name](args, sequence_directory, corruption_name, images, severity)


