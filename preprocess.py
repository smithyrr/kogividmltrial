import os
import subprocess

# Set up paths
data_path = "/home/cognitron/kogivid/UCF1011/UCF-101"
output_path = "/home/cognitron/kogivid/UCF1011/UCF-101-frames"

# Loop through each video in the dataset
for class_name in os.listdir(data_path):
    class_path = os.path.join(data_path, class_name)
    output_class_path = os.path.join(output_path, class_name)
    if not os.path.exists(output_class_path):
        os.makedirs(output_class_path)
    for video_name in os.listdir(class_path):
        video_path = os.path.join(class_path, video_name)
        output_video_path = os.path.join(output_class_path, video_name.split('.')[0])
        if not os.path.exists(output_video_path):
            os.makedirs(output_video_path)
        # Use FFmpeg to extract frames from the video
        subprocess.call(['ffmpeg', '-i', video_path, '-vf', 'fps=1', '-q:v', '2', os.path.join(output_video_path, '%06d.jpg')])
