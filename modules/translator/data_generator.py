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

import copy
import random
from pathlib import Path

import gin
import numpy as np
import numpy.typing as npt
from tensorflow.keras.utils import Sequence

from modules import utils

from . import augmentation


@gin.configurable
class DataGenerator(Sequence):

    def __init__(self, root_folder: str, batch_size: int, labels: dict, n_frames: int):
        root_folder = Path(root_folder)
        self.batch_size = batch_size
        self.n_frames = n_frames

        self.labels_dict = labels
        self.dataset_uuids = list(self.labels_dict.keys())
        self.skeleton_ds = utils.file_utils.load_skeleton_h5(root_folder)

        for k in self.skeleton_ds.keys():
            print(k, len(self.skeleton_ds[k]))

    def __iter__(self):
        return self

    def __len__(self):
        return int(1e4)

    def random_train_sample(self, n_pick: int, hards: list[int], hard_p: float = 0.15) -> npt.ArrayLike:
        """Choosing one video at random from the dataset and sampling data.

        Args:
            n_frames (int): How many frames to sampling from a video.

        Returns:
            npt.ArrayLike: Skeleton sample
        """
        # Hard mining.
        if (hards is not None) and (np.random.random() < hard_p):
            label_idx = np.random.choice(hards)
            random_uuid = self.dataset_uuids[label_idx]
        else:
            random_uuid = random.choice(self.dataset_uuids)
            label_idx = self.labels_dict[random_uuid][0]

        video = random.choice(self.skeleton_ds[random_uuid])

        # Deep copy to avoid data altered from inplace augmentation.
        video = copy.deepcopy(video)

        assert video["n_frames"] > 8

        # Filter visibility.
        video = utils.skeleton_utils.filter_visibility(video)

        # Sampling.
        s_indices = utils.skeleton_utils.random_sampling(video["n_frames"], n_pick)
        video = utils.skeleton_utils.apply_resampling(video, s_indices)

        video = augmentation.augment_video(video)

        return video, label_idx

    def __getitem__(self, item, hards):

        p_batch = np.zeros([self.batch_size, self.n_frames, 15, 3], dtype=np.float32)
        f_batch = np.zeros([self.batch_size, self.n_frames, 25, 3], dtype=np.float32)
        lh_batch = np.zeros([self.batch_size, self.n_frames, 21, 3], dtype=np.float32)
        rh_batch = np.zeros([self.batch_size, self.n_frames, 21, 3], dtype=np.float32)

        y_batch = np.zeros([self.batch_size, 1], dtype=np.int32)

        for i in range(self.batch_size):

            samples, label_idx = self.random_train_sample(n_pick=self.n_frames, hards=hards)

            p_batch[i] = samples["pose_frames"]
            f_batch[i] = samples["face_frames"]
            lh_batch[i] = samples["lh_frames"]
            rh_batch[i] = samples["rh_frames"]

            y_batch[i] = label_idx

        return [p_batch, f_batch, lh_batch, rh_batch], y_batch
