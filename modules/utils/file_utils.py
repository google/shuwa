import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def open_dataset_h5(fpath: Path) -> list[dict]:
    """
	Open dataset file in h5 format.
	each file contain multiple videos.
	"""
    out_data = []
    with h5py.File(fpath, 'r') as f:

        for vid_i in f.keys():
            vid_res = {}
            for k in f[vid_i].keys():
                vid_res[k] = f[vid_i].get(k)[()]
            out_data.append(vid_res)

    return out_data


def write_dataset_h5(fpath: Path, videos: list[dict]):

    fpath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(fpath, 'w') as f:
        for i, res in enumerate(videos):
            dict_group = f.create_group(str(i))

            dict_group.create_dataset("pose_frames", data=res["pose_frames"])
            dict_group.create_dataset("face_frames", data=res["face_frames"])
            dict_group.create_dataset("lh_frames", data=res["lh_frames"])
            dict_group.create_dataset("rh_frames", data=res["rh_frames"])
            dict_group.create_dataset("n_frames", data=res["n_frames"])


def load_skeleton_h5(folder: Path) -> dict:
    all_h5 = [v for v in folder.glob("*.h5")]
    num_h5 = len(all_h5)
    logging.info(f"Globing {folder} Found {num_h5} files.")
    assert num_h5 > 0, f"[ERROR] no .h5 files were found in {folder}."

    kp_database = {}
    # For each h5 file.
    for h5_path in tqdm(all_h5, leave=False):
        kp_database[h5_path.stem] = open_dataset_h5(h5_path)
    return kp_database


def load_latents_npy(folder: Path) -> dict:
    all_npy = [v for v in folder.glob("*.npy")]
    num_npy = len(all_npy)
    logging.info(f"Globing {folder} Found {num_npy} files.")
    assert num_npy > 0, f"[ERROR] no .npy files were found in {folder}."

    database = {}
    # For each h5 file.
    for npy_path in tqdm(all_npy, leave=False):
        database[npy_path.stem] = np.load(npy_path, allow_pickle=True)
    return database
