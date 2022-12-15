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
import copy
import logging
from pathlib import Path

import gin

from modules import translator, utils

gin.parse_config_file('configs/utils.gin')
gin.parse_config_file('configs/translator_inference.gin')

logging.basicConfig(level=logging.DEBUG)


def main(input_dir: Path, min_vid: int):
    translator_manager = translator.TranslatorManager()
    # Load skeleton.
    logging.info(f"Output dir: {translator_manager.knn_dir.as_posix()}")
    logging.info("Loading skeleton h5...")
    skeleton_ds = utils.file_utils.load_skeleton_h5(input_dir)

    # Run feed skeletons to autoencoder.
    for i, (k_name, vid_res_list) in enumerate(skeleton_ds.items()):

        # Encode all videos in the class, without resampling.
        total_vid = len(vid_res_list)

        # Repeat data with augmentations.
        n_repeat = min_vid // total_vid + 1

        # First chunk.
        knn_records = []

        vid_list_clone = copy.deepcopy(vid_res_list)
        for vid_res in vid_list_clone:
            feats = translator_manager.get_feats(vid_res)
            knn_records.append(feats)

        # Augment chunks.
        for _ in range(n_repeat - 1):
            # Deep copy to avoid data altered from inplace augmentation.
            vid_list_clone = copy.deepcopy(vid_res_list)
            for vid_res in vid_list_clone:
                feats = translator_manager.get_feats(vid_res, is_augment=True)
                knn_records.append(feats)

        logging.info(
            f" {str(i)}/{len(skeleton_ds.items())} \t {k_name} \t\t\t total_vid: {total_vid} -> {len(knn_records)} ")

        translator_manager.save_knn_database(k_name, knn_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--min_vid',
                        default=50,
                        type=int,
                        required=False,
                        help="Repeat the dataset till the number exceeds this value.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    main(input_dir, args.min_vid)
