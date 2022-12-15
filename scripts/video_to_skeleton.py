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
import logging
from pathlib import Path

import cv2
import gin
from tqdm import tqdm

from modules import holistic, utils

from . import skeleton_writer

gin.parse_config_file('configs/holistic.gin')

holistic = holistic.HolisticManager()

logging.basicConfig(level=logging.DEBUG)

VIDEO_SIZE = 480


def main(input_dir: Path, out_dir: Path):

    all_folders = [d for d in input_dir.iterdir() if d.is_dir()]
    num_folders = len(all_folders)
    logging.info(f"Found {num_folders} folders.")
    assert num_folders > 0

    # Make output folder
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    skel_writer = skeleton_writer.SkeletonWriter()

    for folder in all_folders:

        # Get all video in folder.
        all_vid_path = [v for v in folder.glob("*.mp4")]
        num_videos = len(all_vid_path)
        logging.info(f"Globing {folder} Found {num_videos} videos.")

        if num_videos < 0:
            logging.warning(f"No video, skipped.")

        # Read each video.
        for video_path in tqdm(all_vid_path):
            try:
                # Check if video valid.
                cap = cv2.VideoCapture(video_path.as_posix())
                ret, frame = cap.read()
                if not ret or frame is None:
                    logging.warning(f"Frame invalid {video_path}, finish video")
                    skel_writer.finish_video()
                    continue

                num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Read each frame.
                frame_control = 0
                while frame_control < num_frame - 1:
                    ret, frame = cap.read()

                    frame_control += 1

                    frame = utils.crop_utils.letterbox_image(frame, VIDEO_SIZE)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect frame with holistic.
                    frame_res = holistic(frame_rgb)

                    skel_writer.add_keypoints(frame_res)

                    cv2.imshow("", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

                    key = cv2.waitKey(1)

                    if key == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                # wrapup video.
                skel_writer.finish_video()
            except Exception as e:
                logging.warning(f"Can't process {video_path}, skipped.")
                skel_writer.reset(clear_dump=False)

        # write h5 contain all video.
        h5_name = folder.name + ".h5"
        skel_writer.finish_file(out_dir / h5_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')

    args = parser.parse_args()

    main(Path(args.input_dir), Path(args.output_dir))
