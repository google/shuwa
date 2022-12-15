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

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from gui import DemoGUI
from modules import utils
from pipeline import Pipeline

cap = cv2.VideoCapture(0)



class Application(DemoGUI, Pipeline):

    def __init__(self):
        super().__init__()
        self.video_loop()

    def show_frame(self, frame_rgb):
        self.frame_rgb_canvas = frame_rgb
        self.update_canvas()

    def tab_btn_cb(self, event):
        super().tab_btn_cb(event)
        # check database before change from record mode to play mode.
        if self.is_play_mode:
            ret = self.translator_manager.load_knn_database()
            if not ret:
                logging.error("KNN Sample is missing. Please record some samples before starting play mode.")
                self.notebook.select(0)

    def record_btn_cb(self):
        super().record_btn_cb()
        if self.is_recording:
            return

        if len(self.pose_history) < 16:
            logging.warnning("Video too short.")
            self.reset_pipeline()
            return

        vid_res = {
            "pose_frames": np.stack(self.pose_history),
            "face_frames": np.stack(self.face_history),
            "lh_frames": np.stack(self.lh_history),
            "rh_frames": np.stack(self.rh_history),
            "n_frames": len(self.pose_history)
        }
        feats = self.translator_manager.get_feats(vid_res)
        self.reset_pipeline()

        # Play mode: run translator.
        if self.is_play_mode:
            res_txt = self.translator_manager.run_knn(feats)
            self.console_box.delete('1.0', 'end')
            self.console_box.insert('end', f"Nearest class: {res_txt}\n")

        # KNN-Record mode: save feats.
        else:
            self.knn_records.append(feats)
            self.num_records_text.set(f"num records: {len(self.knn_records)}")

    def save_btn_cb(self):
        super().save_btn_cb()

        # Read texbox entry, use as folder name.
        gloss_name = self.name_box.get()

        if (gloss_name == ""):
            logging.error("Empty gloss name.")
            return
        if (len(self.knn_records) < 0):
            logging.error("No knn record found.")
            return

        self.translator_manager.save_knn_database(gloss_name, self.knn_records)

        logging.info("database saved.")
        # clear.
        self.knn_records = []
        self.num_records_text.set("num records: " + str(len(self.knn_records)))
        self.name_box.delete(0, 'end')

    def video_loop(self):

        ret, frame = cap.read()
        if not ret:
            logging.error("Camera frame not available.")
            self.close_all()
        frame = utils.crop_utils.crop_square(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t1 = time.time()

        self.update(frame_rgb)

        t2 = time.time() - t1
        cv2.putText(frame_rgb, "{:.0f} ms".format(t2 * 1000), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)
        self.show_frame(frame_rgb)

        self.root.after(1, self.video_loop)

    def close_all(self):

        cap.release()
        cv2.destroyAllWindows()
        sys.exit()


if __name__ == "__main__":
    app = Application()
    app.root.mainloop()
