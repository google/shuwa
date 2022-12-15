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
import glob
import os
import posixpath
import random
import time

import cv2
import numpy as np
from constants import *
from crop_utils import letterbox_image
from tqdm import tqdm
from utils import *

from pipeline import Pipeline

realtime_pipeline = Pipeline()
realtime_pipeline.load_database()

# skip some frame for faster speed.
SKIP_FRAME = 2

KNN_DATASET_PATH = "knn_dataset"


class Application(Pipeline):

    def __init__(self, root_folder):
        self.root_folder = os.path.normpath(root_folder)

        self.sign_folders = glob.glob(os.path.join(root_folder, "*"))

    def process(self):
        for f in self.sign_folders:

            vids_path = glob.glob(os.path.join(f, "*.mp4"))
            random.shuffle(vids_path)

            correct = 0
            correct5 = 0
            fail = []
            assert len(vids_path) > 1
            for v_path in vids_path:
                v_path = os.path.normpath(v_path)

                cap = cv2.VideoCapture(v_path)
                ret, frame = cap.read()
                num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idx = 0

                for frame_idx in range(num_frame - 1):

                    ret, frame = cap.read()
                    if frame_idx % SKIP_FRAME == 0:

                        frame = letterbox_image(frame, 480)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        t1 = time.time()

                        realtime_pipeline.update(frame_rgb)

                        cv2.imshow("frame", frame_rgb)
                        key = cv2.waitKey(1)
                        if key == ord("q"):
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()

                true_name = os.path.basename(os.path.dirname(v_path))
                feats = realtime_pipeline.get_feats(reset=True)
                result_class_name = realtime_pipeline.run_knn(feats)

                if result_class_name == true_name:
                    correct += 1
                # if true_name in top_k_result:
                #     correct5 += 1
                else:
                    fail.append([os.path.split(v_path)[-1], result_class_name])

            print("=" * 10)
            print(true_name)
            print("top1 {}/{}   top5 {}/{}".format(correct, len(vids_path), correct5, len(vids_path)))

            # print("[INFO] Failure caess ", fail)

            log_path = os.path.join(f, "log.txt")
            if os.path.isfile(log_path):
                os.remove(log_path)

            if len(fail) > 0:
                with open(log_path, "w") as outfile:
                    outfile.write("\n".join(str(item) for item in fail))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('root_folder')
    args = parser.parse_args()

    app = Application(args.root_folder)
    app.load_database()
    app.process()
