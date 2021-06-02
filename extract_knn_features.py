# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os, glob
import time
import cv2
import numpy as np


from crop_utils import letterbox_image
from utils import *
from constants import *

from pipeline import Pipeline


sign_pipeline = Pipeline()



def main(mp4_path, output_path_root):
    """Extract 832 knn features from video.
    Alternative from using record mode.

    Args:
        mp4_path ([string]): root directory of mp4 video.
        output_path_root ([string]): output path
    """


    mp4_path = os.path.join(mp4_path, "**","*.mp4")
    all_vid_path = glob.glob(mp4_path)
    
 
  
    for video_path in all_vid_path:   
        video_path = os.path.normpath(video_path)
        
        folder = os.path.join(output_path_root, video_path.split(os.sep)[-2])
        output_path = os.path.join(folder, video_path.split(os.sep)[-1]).replace(".mp4", ".txt")
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        cap = cv2.VideoCapture(video_path)
        print(video_path)
        ret, frame = cap.read()       
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        
        while frame_idx < num_frame-1:    
            
            ret, frame = cap.read()
            frame = letterbox_image(frame, 480)    
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t1 = time.time()

            sign_pipeline.update(frame_rgb)

            cv2.imshow("frame", frame_rgb)
            cv2.waitKey(1)
            frame_idx += 1
            

        feats = sign_pipeline.run_classifier()
        np.savetxt(output_path, feats)
        
        
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mp4_path')
    parser.add_argument('output_path')

    args = parser.parse_args()
    
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)    
       
    
    main(args.mp4_path, args.output_path)
        
        
        
