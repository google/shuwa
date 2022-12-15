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
from pathlib import Path

import numpy as np

from modules import utils

MINIMUM_FRAMES = 12


class SkeletonWriter():

    def __init__(self):
        self.reset(clear_dump=True)

    def reset(self, clear_dump: bool):
        self.pose_buffer = []
        self.face_buffer = []
        self.lh_buffer = []
        self.rh_buffer = []
        if clear_dump:
            self.dump_list = []

    def add_keypoints(self, frame_res: dict):
        # Pose.
        assert frame_res["pose_4d"].shape == (15, 4)
        self.pose_buffer.append(frame_res["pose_4d"])

        # Face.
        assert frame_res["face_3d"].shape == (25, 3)
        self.face_buffer.append(frame_res["face_3d"])

        # Left hand.
        assert frame_res["lh_3d"].shape == (21, 3)
        self.lh_buffer.append(frame_res["lh_3d"])

        # Right hand.
        assert frame_res["rh_3d"].shape == (21, 3)
        self.rh_buffer.append(frame_res["rh_3d"])

    def finish_video(self):

        if len(self.pose_buffer) > MINIMUM_FRAMES:

            n_frames = len(self.pose_buffer)
            vid_res = {
                "pose_frames": np.stack(self.pose_buffer),
                "face_frames": np.stack(self.face_buffer),
                "lh_frames": np.stack(self.lh_buffer),
                "rh_frames": np.stack(self.rh_buffer),
                "n_frames": n_frames
            }

            self.dump_list.append(vid_res)
        else:
            logging.warning("Video too short, skipped.")

        self.reset(clear_dump=False)

    def finish_file(self, h5_path: Path):
        if self.dump_list == []:
            return

        to_dump = []

        # Merge to old skeleton file, if exist.
        if h5_path.is_file():
            logging.info(f"Found old file at {h5_path}, merging new videos to the list.")
            old_list = utils.file_utils.open_dataset_h5(h5_path)
            to_dump.extend(old_list)

        to_dump.extend(self.dump_list)
        utils.file_utils.write_dataset_h5(h5_path, to_dump)
        self.reset(clear_dump=True)
        logging.info(f"Write output file at {h5_path}.")
