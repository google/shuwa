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

import numpy as np
from constants import *


def build_part_with_coords(score_threshold, local_max_radius, scores, offsets, num_keypoints=11, output_stride=16):
        

    parts = np.array([0,0,np.array([]), np.array([])])
    for keypoint_id in range(0, num_keypoints):
        kp_scores = scores[:, :, keypoint_id].copy()
        kp_scores[kp_scores < score_threshold] = 0.           
        y, x = np.unravel_index(kp_scores.argmax(), kp_scores.shape)
        heatmap_coords = np.array([y, x])
        image_coords = heatmap_coords * output_stride + offsets[y, x, keypoint_id]   
        parts = np.vstack([parts, [scores[y, x, keypoint_id], keypoint_id, heatmap_coords, image_coords.astype(np.float32)]])
                         

    return parts[1:]

    
    
    
def decode_single_pose(
        heatmap, offsets, displacements_fwd, displacements_bwd, output_stride,
        max_pose_detections=1, score_threshold=0.35, nms_radius=20, min_pose_score=0.5):

    pose_scores = np.zeros(1)
    pose_keypoint_scores = np.zeros((1, NUM_KEYPOINTS))
    pose_keypoint_coords = np.ones((1, NUM_KEYPOINTS, 2))*IGNORE_VALUE
    pose_keypoint_heatmap_coords = np.ones((1, NUM_KEYPOINTS, 2))*IGNORE_VALUE

    height = heatmap.shape[0]
    width = heatmap.shape[1]
    offsets = offsets.reshape(height, width, 2, -1).swapaxes(2, 3)

    
    scored_parts = build_part_with_coords(score_threshold, 17, heatmap, offsets,
                                          num_keypoints=NUM_KEYPOINTS)
    
    # exit if can't find any part.
    if len(scored_parts.shape) <= 1:
        return pose_scores, pose_keypoint_scores, pose_keypoint_coords
        
    
    scored_parts = sorted(scored_parts, key=lambda x: x[1], reverse=False)
    
    # exit if can't find nose (we need face to perform sign language).
    if scored_parts[0][1] != 0:
        return pose_scores, pose_keypoint_scores, pose_keypoint_coords
    

    for current_idx in range(0, NUM_KEYPOINTS):
        
        # pick candidates from heatmap.
        candidates = []
        for part in scored_parts:
            if part[1] == current_idx:
                candidates.append(part) 
                                
                    

        if len(candidates) == 1:                         
            final_score, _, final_heatmap_coords, final_image_coords = candidates[0] 
            pose_keypoint_scores[0, current_idx] = final_score
            pose_keypoint_heatmap_coords[0, current_idx, :] = final_heatmap_coords     
            pose_keypoint_coords[0, current_idx, :] = final_image_coords
            
           
 
    pose_scores[0] = 1.      
    return pose_scores, pose_keypoint_scores, pose_keypoint_coords