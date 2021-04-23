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

import sys, os
import glob
import time
import cv2
from datetime import datetime
import numpy as np
from crop_utils import crop_square
from constants import *
from gui import DemoGUI
from pipeline import Pipeline


KNN_DATASET_PATH = "knn_dataset"
cap = cv2.VideoCapture(0)

class Application(DemoGUI, Pipeline):

    def __init__(self):
        super().__init__()
        
        self.result_class_name = ""     
        self.database = []
        self.labels = []
        self.records = []      
        self.video_loop()


    def load_database(self):
        print("[INFO] Reading database...")
        self.database = []
        self.labels= []
        all_folder = glob.glob(os.path.join(KNN_DATASET_PATH, "*"))
           
        for folder in all_folder:
            all_file = glob.glob(os.path.join(folder, "*.txt"))
            class_label = os.path.split(folder)[-1]
            for f in all_file:
                self.database.append(np.loadtxt(f))
                self.labels.append(class_label)
                
        if len(self.database) == 0:
            print("[INFO] Can't load any database file. Try record any database first.")           
            self.notebook.select(0)
            return False
        
        else:    
            self.database = np.stack(self.database)
            self.labels = np.array(self.labels)       
            print("[INFO] Found ", len(all_folder), "classes.")    
            self.result_class_name = ""    
            return True

        

    def show_frame(self, frame_rgb):
        self.frame_rgb_canvas = frame_rgb
        self.update_canvas()
    

          
    def change_mode_on_tab(self, event):         
        super().change_mode_on_tab(event)
        # check database before change from record mode to play mode.
        if self.is_play_mode:
            self.load_database()        
                        
                            
    def toggle_record_button(self):
        super().toggle_record_button()    
        if not self.is_recording:
            if len(self.pose_history) > NUM_FRAME_SAMPLES:
                # playmode
                if self.is_play_mode:
                    result_class_name = self.run_knn_classifier()
                    self.console_box.delete('1.0', 'end')
                    self.console_box.insert('end',
                        "Nearest class: {:s}\n".format(result_class_name))

                        
                # record mode.
                else:
                    # add video track.                
                    self.records.append(self.run_classifier())
                    self.num_records_text.set("num records: "+ str(len(self.records)))
                    
            else:
                print("[ERROR] Video too short.")
                
        self.reset_pipeline()
        
        
                
 
    def save_database(self):      
        # Save recorded templates to files.           
        super().save_database()        
        timestamp =  datetime.now().strftime("%d%m%Y%H%M%S")
        # Read texbox entry, use as folder name.
        if self.name_box.get() != "" and len(self.records) > 0:                
            folder_name = self.name_box.get()            
            folder_path = os.path.join(KNN_DATASET_PATH, folder_name)
                    
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            for i, a in enumerate(self.records):                
                np.savetxt(folder_path+os.sep+timestamp+str(i)+".txt", a, fmt='%.8f')
            print("[INFO] database saved.")
            # clear.
            self.records = []
            self.num_records_text.set("num records: "+ str(len(self.records)))            
            self.name_box.delete(0, 'end')            
                        
            
        

    def video_loop(self):

        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera frame not available.")
            self.close_all()
        frame = crop_square(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        t1 = time.time()

        self.update(frame_rgb)


        t2 = time.time() - t1
        cv2.putText(frame_rgb, "{:.0f} ms".format(t2*1000), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)
        self.show_frame(frame_rgb)        

        self.root.after(1, self.video_loop)
        
        
    def close_all(self):           
            
        cap.release()           
        cv2.destroyAllWindows()       
        sys.exit()
        
        
if __name__ == "__main__":
    app = Application()       
    app.root.mainloop()