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

import sys, os, glob
import time
import cv2

sys.path.insert(1, 'posenet')
sys.path.insert(1, 'face_landmark')
sys.path.insert(1, 'hand_landmark')
sys.path.insert(1, 'classifier')

from pose_manager import PoseManager
from face_manager import FaceManager
from hand_pipeline import HandPipeline
from crop_utils import letterbox_image
import argparse
from utils import *
from annotations_maker import AnnotationsMaker
from constants import *


pose_manager = PoseManager()
face_manager = FaceManager()
hand_pipeline = HandPipeline()



def main(mp4_path, output_path_root):


    mp4_path = os.path.join(mp4_path, "**", "*.mp4")
    all_vid_path = glob.glob(mp4_path)

    for video_path in all_vid_path:   
        video_path = os.path.normpath(video_path)
        
        folder = os.path.join(output_path_root, video_path.split(os.sep)[-2]) 
        output_path = os.path.join(folder, video_path.split(os.sep)[-1])        
            
        annotations_maker = AnnotationsMaker(output_path)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
       

        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_control = 0

        fail = 0
        while frame_control < num_frame-1:    
            
            ret, frame = cap.read()

            frame_control += 1
                                
            if frame_control%2 != 0:     
                continue                                
        
            frame = letterbox_image(frame, 480)    
            video_height, video_width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t1 = time.time()


            # ─── POSENET ────────────────────────────────────────────────────────────────────
            
            pose_scores, keypoint_scores, poses_keypoints = pose_manager(frame_rgb)
            if pose_scores < POSE_THRESHOLD:
                print('[WARN] No person. skip this video.')
                fail = 1
                break            
            
            
            # ─── FACE-LANDMARK ───────────────────────────────────────────────────────────────            
            # Crop face region from posenet keypoints.
            face_rect = get_face_rect_from_posenet(poses_keypoints)
            face_rgb = crop_from_rect(frame_rgb, face_rect)
            
            # Draw face box.
            cv2.drawContours(frame_rgb, [cv2.boxPoints(face_rect).astype("int")], 0, (0, 255, 0), 2)
            
            # Predict face keypoints.
            face_flag, local_face_keypoints = face_manager(face_rgb)
            
            if face_flag < FACE_THRESHOLD:                
                print('[WARN] Face not found. skip this frame.')
                continue
            local_face_keypoints = local_face_keypoints[:, :FACE_JOINT_DIMS]
            

            # Convert local keypoints to global keypoints.
            global_face_keypoints = local2global_keypoints(local_face_keypoints, face_rect)
            

            # ─── HAND-LANDMARK ───────────────────────────────────────────────────────────────    
            hand_rect, global_hand_keypoints = hand_pipeline(frame_rgb, poses_keypoints, keypoint_scores)




            # ─── DRAW ───────────────────────────────────────────────────────────────────────    
            pose_manager.draw_keypoints(frame_rgb, pose_scores, keypoint_scores, poses_keypoints)
            face_manager.draw_keypoints(frame_rgb, global_face_keypoints)
            hand_pipeline.draw(frame_rgb, hand_rect, global_hand_keypoints)
             
             
            # save in [0, 1] space.            
            global_hand_keypoints[0] /= video_height
            global_hand_keypoints[1] /= video_height            
            annotations_maker.add_keypoints(poses_keypoints[0].copy()/video_height, global_face_keypoints.copy()/video_height, global_hand_keypoints.copy())
        
        
        
        
            t2 = time.time() - t1
            cv2.putText(frame_rgb,"frame_time: {:.0f} ms".format(t2 * 1000), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("posenet", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))                           
            # cv2.imwrite(output_path.replace('.mp4', '_'+str(frame_control)+'.jpg'), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))           
            

            key = cv2.waitKey(1)
            

            if key == ord("q"):      
                cap.release()
                cv2.destroyAllWindows()
                break
        
        if not fail:
            annotations_maker.finish_file()
            
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mp4_path')
    parser.add_argument('output_path')

    args = parser.parse_args()
    
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)    
        os.makedirs(os.path.join(args.output_path, "train"))
        os.makedirs(os.path.join(args.output_path, "val"))
        
        
    
    main(args.mp4_path, args.output_path)
        
        
        
