import argparse
import subprocess
from pathlib import Path

import gin

from modules import translator

gin.parse_config_file('configs/translator_inference.gin')
gin.parse_config_file('configs/utils.gin')


def main(out_dir, checkpoint):

    # ---------------------------- SAVE FULL h5 MODEL ---------------------------- #
    model = translator.get_model()
    model.load_weights(checkpoint)
    batch_size = model.outputs[0].shape[0]
    n_feats = model.outputs[0].shape[1]
    n_classes = model.outputs[1].shape[1]
    print("batch_size:", batch_size)
    print("n_feats:", n_feats)
    print("n_classes:", n_classes)
    assert batch_size == 1

    h5_out_path = out_dir / "model.h5"
    model.save(h5_out_path)

    # ----------------------- CONVERT FULL H5 TO TFJS MODEL ---------------------- #

    result = subprocess.run(["tensorflowjs_converter", "--input_format=keras", f"{h5_out_path}", f"{out_dir}"],
                            capture_output=True,
                            text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',
                        default="scripts/one_click/tfjs_model",
                        required=False,
                        help="Output tfjs model path.")
    parser.add_argument('--checkpoint', default=None, required=True, help="Continue training from checkpoint.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    main(out_dir, args.checkpoint)
