import numpy as np
import os, pickle
from constants import *
from pathlib import Path

class AnnotationsMaker():
    def __init__(self, output_path):        
        # save folder path.
        root_path = os.path.join(*output_path.split(os.sep)[:-2])
        folder_name = output_path.split(os.sep)[-2]
        
        # save file name.
        file_name = output_path.split(os.sep)[-1]
        file_name = file_name.replace('.mp4', '.pkl')
        
        # this annotation will go to train or val set.    
        train_val = "val" if np.random.random() < SPLIT_TRAIN_VAL else "train"            
        
        folder_path = os.path.join(root_path, train_val, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)   
        self.output_path =os.path.join(folder_path, file_name)                         

        self.posenet_holder = []
        self.face_holder = []
        self.left_hand_holder = []
        self.right_hand_holder = []
        

    def add_keypoints(self, posenet_keypoints, global_face_keypoints, hand_keypoints):      
        # posenet.
        self.posenet_holder.append(posenet_keypoints)        
     
        # face.     
        self.face_holder.append(global_face_keypoints)        
        
        # hand.    
        self.left_hand_holder.append(hand_keypoints[0])     
        self.right_hand_holder.append(hand_keypoints[1])
        
       
    def finish_file(self):
        if len(self.posenet_holder) > 12:

            self.posenet_holder = np.stack(self.posenet_holder)
            self.face_holder = np.stack(self.face_holder)
            self.left_hand_holder = np.stack(self.left_hand_holder)
            self.right_hand_holder = np.stack(self.right_hand_holder)

            print(self.output_path, len(self.posenet_holder))
            # save list as pkl file.
            
            Path(os.path.split(self.output_path)[0]).mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'wb') as handle:
                pickle.dump([self.posenet_holder, self.face_holder, self.left_hand_holder, self.right_hand_holder], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
