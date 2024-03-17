# Benchmarking the Robustness of UAV Tracking Against Common Corruptions

## Introduction
This repository presents a benchmark study focused on the robustness of UAV tracking methods against various common corruptions. Tracking robustness is crucial for practical UAV applications, where visual conditions can significantly vary due to environmental factors.

![Corruption Types](car_zoom_blur_video.mp4)

## How to use this repository

1. **Download Datasets**:
   - Obtain the [**DTB70**](https://github.com/flyers/drone-tracking) and [**UAV123_10fps**](https://cemse.kaust.edu.sa/ivul/uav123) datasets, which serve as the clean foundation for corruption simulation. 

2. **Dataset Setup**:
   - Place the downloaded datasets in a designated directory within this repository. It's recommended to organize the datasets in a structured manner, such as:
     ```
     /data/DTB70
     /data/UAV123_10fps
     ```

### Generating Depth Maps

1. **Run Depth Map Generator**:
   - Execute the depth map generation script provided in the `runs/` directory. Adjust the script's parameters if necessary to point to your dataset locations.
   - Command for generate depth map for DTB70:
     ```bash
     python main.py --phase 'depth_estimation' --seq_dir '/yourpath/DTB70'
     ```
   - Command for generate depth map for UAV123_10fps:
     ```bash
     python main.py --phase 'depth_estimation' --seq_dir '/yourpath/UAV123_10fps'
     ```
### Download libaray Corruptions from ImageNet-C
- Command for install libaray from ImageNet-C:
```bash
git clone https://github.com/hendrycks/robustness.git
cd robustness/ImageNet-C/imagenet_c/imagenet_c
cd ..
pip install -e .
```

### Applying Corruptions

1. **Select Corruption Types**:
   - This benchmark includes 17 corruption types, each with three severity levels(1 3 5).  'runs/generator' by default generate 17 types of corruption together, you could modify corruption types if necessary.

2. **Generate Corruptions**:
   - Example command for DTB70:
     ```bash
     python main.py --seq_dir /data/DTB70 --output_dir /data/UAV123_10fps_noisy 
     ```
   - Example command for UAV123_10fps:
     ```bash
     python main.py --seq_dir /data/UAV123_10fps --output_dir /data/UAV123_10fps_noisy 
     ```

## Quantitative Results
We evaluated 12 trackers on the UAV-C benchmark. The table below summarizes the performance of each tracker across all types of corruptions. The last row is measured with the metric \( mS_{cor} \), averaged over all corruption types.
![Corruption Types](Performances.png)
![Success](radar_success.png)
![Precision](radar_precision.png)
