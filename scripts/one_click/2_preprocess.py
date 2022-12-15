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
import json
import os
from datetime import datetime
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage


def main(out_dir, master_path, credentials):

    # ---------------  CONNECT TO FIREBASE AND GET ALL VIDEOS LIST --------------- #

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials
    firebase_admin.initialize_app()

    # Initialise a client
    storage_client = storage.Client("bit-sign-language")

    db = firestore.client()

    keys = ["signId", "status", "signLanguage", "consent", "bucketId", "videoObjectId"]
    docs = db.collection("prod/data-collection-service/collected_sign_videos").select(keys).where(
        "status", "==", "accepted").stream()
    #.where("signLanguage", "==", lang)\

    d_docs = [doc.to_dict() for doc in docs]

    # ---------------------------  OPEN MASTER JSON --------------------------- #
    with open(master_path, "r") as f:
        master_dict = json.load(f)

    master_vids = []
    for k, v in master_dict.items():
        master_vids.extend(v)

    # ----------------------------  DOWNLOAD NEW VIDEOS --------------------------- #
    new_master = {}
    out_folder = Path(out_dir)
    for data in d_docs:

        video_name = Path(data["videoObjectId"]).name

        # if new videos.
        if video_name in master_vids:
            continue

        out_file_path = out_folder / data["signId"] / video_name
        if out_file_path.exists():
            continue

        # out_file_path.parent.mkdir(parents=True, exist_ok=True)
        # bucket = storage_client.get_bucket(data["bucketId"])
        # blob = bucket.blob(data["videoObjectId"])
        # blob.download_to_filename(str(out_file_path))
        print(out_file_path.as_posix())

        if data["signId"] in new_master:
            new_master[data["signId"]].append(video_name)
        else:
            new_master[data["signId"]] = [video_name]

    # ---------------------------- UPDATE MASTER JSON ---------------------------- #
    master_dict.update(new_master)
    with open(master_path, "w") as outfile:
        json.dump(master_dict, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', default="scripts/one_click/download", type=str, help="download output dir.")

    parser.add_argument('--master_json',
                        default="scripts/one_click/master.json",
                        type=str,
                        help="master json contain all video filename.")
    parser.add_argument('--credentials',
                        default="scripts/one_click/bit-sign-language-028ca8a3c70a.json",
                        type=str,
                        help="google credentials.")

    args = parser.parse_args()

    main(args.out_dir, args.master_json, args.credentials)
