import pandas as pd
import json
import numpy
import os
import os.path as path
import pdb
from moviepy.video.io.VideoFileClip import VideoFileClip
import time
from PIL import Image
import torchvision


# narrations = pd.read_json('/shared/mandi/ego4d_data/v1/annotations/narration.json')
# narrations['ad4f61f0-4c0f-4ce1-bdaa-e57b79250527']['narration_pass_1']['narrations'][1]
# narrations = json.load('/shared/mandi/ego4d_data/v1/annotations/narration.json')
# pdb.set_trace()
ego4d_dir = '/shared/mandi/ego4d_data/v1/clips'
outputego4d_dir = '/shared/ademi_adeniji/datasets/ego4d'
for video in os.listdir(ego4d_dir):
    video_dir = path.join(ego4d_dir, video)
    if not video.endswith('.mp4'):
        continue
    frames = VideoFileClip(video_dir, audio=False)
    clip_dir = path.join(outputego4d_dir, video[:-4])
    os.makedirs(clip_dir, exist_ok=True)
    resize_image = torchvision.transforms.Resize([256, 256])
    center_crop = torchvision.transforms.CenterCrop([224, 224])
    print(f"processing video {video}")
    
    start = time.time()
    for (i, frame) in enumerate(frames.iter_frames()):
        frame = Image.fromarray(frame)
        frame = center_crop(resize_image(frame))
        frame.save(clip_dir + '/' + str(i) + '.png')
    end = time.time()
    duration = end - start
    print(f"processed video in {duration} seconds")

