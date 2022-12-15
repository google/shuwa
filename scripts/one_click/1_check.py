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


def main(master_json, credentials):

    # ---------------- 1.1 CONNECT TO FIREBASE AND GET ALL VIDEOS LIST --------------- #

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

    # --------------------------- 1.2 OPEN MASTER JSON --------------------------- #
    with open(master_json, "r") as f:
        master_json = json.load(f)

    master_vids = []
    for k, v in master_json.items():
        master_vids.extend(v)

    # --------------------------- 1.3 CHECK FOR NEW VIDEOS --------------------------- #
    new_stats = {}

    total_new_vids = 0
    for data in d_docs:

        video_name = Path(data["videoObjectId"]).name
        # if new videos.
        if not video_name in master_vids:
            total_new_vids += 1

            if data["signId"] in new_stats:
                new_stats[data["signId"]] += 1

            else:
                new_stats[data["signId"]] = 0

    print("total new videos", total_new_vids)

    for gloss, v in new_stats.items():
        if gloss not in master_json:
            print("*", end="")
        print(gloss, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_json',
                        default="scripts/one_click/master.json",
                        type=str,
                        help="master json contain all video filename.")
    parser.add_argument('--credentials',
                        default="scripts/one_click/bit-sign-language-028ca8a3c70a.json",
                        type=str,
                        help="google credentials.")
    args = parser.parse_args()

    main(args.master_json, args.credentials)
