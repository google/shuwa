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

import gin
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from modules import utils

from . import augmentation, model


@gin.configurable
class TranslatorManager():

    def __init__(self, model_path: str, labels: dict, knn_dir: str, n_frames: int) -> None:
        self.n_frames = n_frames
        self.softmax_labels = labels
        self.knn_dir = Path(knn_dir)
        self.knn_dir.mkdir(parents=True, exist_ok=True)

        self.model = model.get_model()
        self.model.load_weights(model_path)
        self.model = tf.function(self.model)

        self.knn_database = []
        self.knn_labels = []

    def load_knn_database(self):
        logging.info("Reading database...")

        txt_files = list(Path(self.knn_dir).glob("*.txt"))

        if len(txt_files) == 0:
            return False

        knn_database = []
        knn_labels = []

        for txt in txt_files:
            arr = np.loadtxt(txt)
            knn_database.extend(arr)
            knn_labels.extend([txt.stem] * len(arr))

        self.knn_database = np.stack(knn_database)
        self.knn_labels = np.array(knn_labels)

        return True

    def save_knn_database(self, gloss_name, knn_records):
        output_path = Path(self.knn_dir, gloss_name + ".txt")

        knn_feats = []
        if output_path.is_file():
            old_feats = np.loadtxt(output_path)
            knn_feats.extend(old_feats)

        knn_feats.extend(knn_records)
        np.savetxt(output_path, knn_feats, fmt='%.8f')

    def preprocess_input(self, vid_res: dict, resampling: int):
        # Remove non-visible joints.
        vid_res = utils.skeleton_utils.filter_visibility(vid_res)
        if resampling > 0:
            indices = utils.skeleton_utils.uniform_sampling(vid_res["n_frames"], n_pick=resampling)
            vid_res["n_frames"] = resampling
            vid_res = utils.skeleton_utils.apply_resampling(vid_res, indices)

        return vid_res

    def get_feats(self, vid_res: dict, is_augment=False):
        vid_res = self.preprocess_input(vid_res, self.n_frames)

        if is_augment:
            vid_res = augmentation.augment_video(vid_res)

        feats_out, cls_out = self.model([
            vid_res["pose_frames"][np.newaxis], vid_res["face_frames"][np.newaxis], vid_res["lh_frames"][np.newaxis],
            vid_res["rh_frames"][np.newaxis]
        ])
        return feats_out.numpy().squeeze()

    def run_knn(self, feats: npt.ArrayLike, k=5):

        dists = np.square(self.knn_database - feats)
        dists = np.sqrt(np.sum(dists, axis=-1))

        # top k nearst samples.
        top_indices = np.argsort(dists)[:k]
        top_lables = self.knn_labels[top_indices]

        # mode.
        vals, counts = np.unique(top_lables, return_counts=True)
        index = np.argmax(counts)
        res_txt = vals[index]

        return res_txt
