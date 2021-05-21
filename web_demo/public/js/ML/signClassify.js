/**
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  decodePose,
  local2GlobalKeypoints,
  getRadians,
  getDistance,
} from "./utility.js";
import {
  SELECTED_POSENET_JOINTS,
  SELECTED_FACE_POINTS,
  LABELS,
} from "./const.js";

export default class SignLanguageClassifyModel {
  constructor() {
    this.poseModel;
    this.faceMeshModel;
    this.handPoseModel;
    this.classifyModel;
    this.faceObj = {
      leftEarKeypoint: [0, 0],
      rightEarKeypoint: [0, 0],
      radians: 0,
      center: 0,
      distance: 0,
      imageBuffer: 0,
      result: [],
    };

    this.leftHand = {
      writsKeypoint: [0, 0],
      midfinKeypoint: [0, 0],
      radians: 0,
      center: 0,
      distance: 0,
      imageBuffer: 0,
      result: [],
      isFound: true,
    };
    this.rightHand = {
      writsKeypoint: [0, 0],
      midfinKeypoint: [0, 0],
      radians: 0,
      center: 0,
      distance: 0,
      imageBuffer: 0,
      result: [],
      isFound: true,
    };
  }

  async initModel() {
    return new Promise(async (resolve) => {
      // init poseModel
      this.poseModel = await tf.loadGraphModel("./model/midfin/model.json");
      await this.poseModel.predict(tf.zeros([1, 257, 257, 3]));
      console.log(`Pose model loaded`);
      // init faceMeshModel
      this.faceMeshModel = await tf.loadGraphModel(
        "./model/face/js_f16/model.json"
      );
      await this.faceMeshModel.predict(tf.zeros([1, 192, 192, 3]));
      console.log(`Face model loaded`);
      // init handPoseModel
      this.handPoseModel = await tf.loadGraphModel(
        "./model/hand/js_f16/model.json"
      );
      await this.handPoseModel.predict(tf.zeros([1, 256, 256, 3]));
      console.log(`hand model loaded`);
      // init classifyModel
      this.classifyModel = await tf.loadGraphModel(
        "./model/classification/phase_1/model.json"
      );

      const poseArr = tf.zeros([1, 16, 13, 2]);
      const faceArr = tf.zeros([1, 16, 24, 2]);
      const handLeftArr = tf.zeros([1, 16, 21, 2]);
      const handRightArr = tf.zeros([1, 16, 21, 2]);

      const wordResult = await this.classifyModel.predict({
        pose_frames_input: poseArr,
        face_frames_input: faceArr,
        left_hand_frames_input: handLeftArr,
        right_hand_frames_input: handRightArr,
      });

      console.log(wordResult);
      const resultArray = wordResult[1].dataSync();
      console.log(resultArray);
      console.log(`classify model loaded`);
      console.log(LABELS[resultArray.indexOf(Math.max(...resultArray))]);
      // console.log(wordResult.dataSync().indexOf(Math.max(...resultArray)));
      resolve(true);
    });
  }

  async initPose() {
    return new Promise(async (resolve) => {
      // init poseModel
      this.poseModel = await tf.loadGraphModel("./model/midfin/model.json");
      await this.poseModel.predict(tf.zeros([1, 257, 257, 3]));
      console.log(`Pose model loaded`);
      resolve(true);
    });
  }

  async initFace() {
    return new Promise(async (resolve) => {
      this.faceMeshModel = await tf.loadGraphModel(
        "./model/face/js_f16/model.json"
      );
      await this.faceMeshModel.predict(tf.zeros([1, 192, 192, 3]));
      console.log(`Face model loaded`);
      resolve(true);
    });
  }

  async initHand() {
    return new Promise(async (resolve) => {
      this.handPoseModel = await tf.loadGraphModel(
        "./model/hand/js_f16/model.json"
      );
      await this.handPoseModel.predict(tf.zeros([1, 256, 256, 3]));
      console.log(`hand model loaded`);
      resolve(true);
    });
  }

  async initClassify() {
    return new Promise(async (resolve) => {
      this.classifyModel = await tf.loadGraphModel(
        "./model/classification/phase_1/model.json"
      );

      const poseArr = tf.zeros([1, 16, 13, 2]);
      const faceArr = tf.zeros([1, 16, 24, 2]);
      const handLeftArr = tf.zeros([1, 16, 21, 2]);
      const handRightArr = tf.zeros([1, 16, 21, 2]);

      await this.classifyModel.predict({
        pose_frames_input: poseArr,
        face_frames_input: faceArr,
        left_hand_frames_input: handLeftArr,
        right_hand_frames_input: handRightArr,
      });
      console.log(`classify model loaded`);
      resolve(true);
    });
  }

  // image stack prediction
  async predict(imagestack) {
    return new Promise(async (resolve) => {
      tf.engine().startScope();
      tf.env().set("WEBGL_PACK_DEPTHWISECONV", true);
      tf.env().set("WEBGL_FLUSH_THRESHOLD", 1);

      // const imageTensorForPredictionStack = []; // this for batch prediction
      let timeMs = 0;
      const poseStacks = [];
      const leftHandStacks = [];
      const rightHandStacks = [];
      const faceStacks = [];
      // for loop in imagestack
      for (const image of imagestack) {
        const tic = Date.now();
        console.log("image: ", image);
        const imageInputTensor = tf
          .slice(
            tf.tensor3d(Array.from(image.data), [257, 257, 4]),
            [0, 0, 0],
            [257, 257, 3]
          )
          .toFloat()
          .expandDims(0);
        const inputTensor = imageInputTensor.div(127.5).sub(1);
        // imageTensorForPredictionStack.push(inputTensor.dataSync()); // this for batch predictioon

        // if test batch, comment this --
        const poseResult = await this.poseModel?.predict({
          input: inputTensor,
        });
        inputTensor.dispose();

        console.log(poseResult);
        const POSE = await decodePose(
          poseResult[0].squeeze(0),
          poseResult[1].squeeze(0),
          16
        );
        console.log(POSE);
        // -- to this

        const poseResultKeypoints = [];
        for (let i = 0, n = POSE.keypoints.length; i < n; i += 1) {
          if (SELECTED_POSENET_JOINTS.includes(POSE.keypoints[i].part)) {
            // console.log(pose.keypoints[i].part, ' index : ', i)
            let THRESHOULD = 0.35;
            if (POSE.keypoints[i].score < THRESHOULD) {
              POSE.keypoints[i].position.x = 0;
              POSE.keypoints[i].position.y = 0;
            }
            poseResultKeypoints.push([
              POSE.keypoints[i].position.x,
              POSE.keypoints[i].position.y,
            ]);
            if (POSE.keypoints[i].part === "leftWrist") {
              if (POSE.keypoints[i].score < THRESHOULD) {
                this.leftHand.isFound = false;
              } else {
                this.leftHand.isFound = true;
                this.leftHand.writsKeypoint = [
                  POSE.keypoints[i].position.x,
                  POSE.keypoints[i].position.y,
                ];
              }
            }

            if (POSE.keypoints[i].part === "rightWrist") {
              if (POSE.keypoints[i].score < THRESHOULD) {
                this.rightHand.isFound = false;
              } else {
                this.rightHand.isFound = true;
                this.rightHand.writsKeypoint = [
                  POSE.keypoints[i].position.x,
                  POSE.keypoints[i].position.y,
                ];
              }
            }

            if (POSE.keypoints[i].part === "leftMidfin") {
              if (POSE.keypoints[i].score < THRESHOULD) {
                this.leftHand.isFound = false;
              } else {
                this.leftHand.isFound = true;
                this.leftHand.midfinKeypoint = [
                  POSE.keypoints[i].position.x,
                  POSE.keypoints[i].position.y,
                ];
              }
            }

            if (POSE.keypoints[i].part === "rightMidfin") {
              if (POSE.keypoints[i].score < THRESHOULD) {
                this.rightHand.isFound = false;
              } else {
                this.rightHand.isFound = true;
                this.rightHand.midfinKeypoint = [
                  POSE.keypoints[i].position.x,
                  POSE.keypoints[i].position.y,
                ];
              }
            }

            if (POSE.keypoints[i].part === "rightEar") {
              this.faceObj.rightEarKeypoint = [
                POSE.keypoints[i].position.x,
                POSE.keypoints[i].position.y,
              ];
            }
            if (POSE.keypoints[i].part === "leftEar") {
              this.faceObj.leftEarKeypoint = [
                POSE.keypoints[i].position.x,
                POSE.keypoints[i].position.y,
              ];
            }
          }
        }
        this.faceObj.radians = await getRadians(
          this.faceObj.leftEarKeypoint,
          this.faceObj.rightEarKeypoint,
          1
        );
        this.leftHand.radians = await getRadians(
          this.leftHand.writsKeypoint,
          this.leftHand.midfinKeypoint,
          2
        );
        this.rightHand.radians = await getRadians(
          this.rightHand.writsKeypoint,
          this.rightHand.midfinKeypoint,
          2
        );

        this.faceObj.center = [
          (this.faceObj.rightEarKeypoint[0] + this.faceObj.leftEarKeypoint[0]) /
            2,
          (this.faceObj.rightEarKeypoint[1] + this.faceObj.leftEarKeypoint[1]) /
            2,
        ];
        this.faceObj.distance = await getDistance(
          this.faceObj.leftEarKeypoint,
          this.faceObj.rightEarKeypoint
        );

        this.leftHand.center = this.leftHand.midfinKeypoint;
        this.leftHand.distance = await getDistance(
          this.leftHand.writsKeypoint,
          this.leftHand.midfinKeypoint
        );

        this.rightHand.center = this.rightHand.midfinKeypoint;
        this.rightHand.distance = await getDistance(
          this.rightHand.writsKeypoint,
          this.rightHand.midfinKeypoint
        );

        const leftHandResult = await this.handPrediction(
          this.leftHand.radians,
          this.leftHand.distance,
          [this.leftHand.center[0] / 257, this.leftHand.center[1] / 257],
          imageInputTensor
        );

        if (this.leftHand.isFound) {
          this.leftHand.result = leftHandResult[1];
        } else if (!this.leftHand.isFound) {
          this.leftHand.result = tf.zeros([21, 2]);
        }

        const rightHandResult = await this.handPrediction(
          this.rightHand.radians,
          this.rightHand.distance,
          [this.rightHand.center[0] / 257, this.rightHand.center[1] / 257],
          imageInputTensor
        );
        // rightHand.imageBuffer = rightHandResult[0]

        if (this.rightHand.isFound) {
          this.rightHand.result = rightHandResult[1];
        } else if (!this.rightHand.isFound) {
          this.rightHand.result = tf.zeros([21, 2]);
        }

        const input_tensor = imageInputTensor;
        const faceRotatedImage = tf.image.rotateWithOffset(
          input_tensor,
          this.faceObj.radians,
          0,
          [this.faceObj.center[0] / 257, this.faceObj.center[1] / 257]
        );
        input_tensor.dispose();
        tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 100 * 1024 * 1024);
        const cropImage = tf.image.cropAndResize(
          faceRotatedImage,
          [
            [
              this.faceObj.center[1] / 257 - this.faceObj.distance / 257,
              this.faceObj.center[0] / 257 - this.faceObj.distance / 257,
              this.faceObj.center[1] / 257 + this.faceObj.distance / 257,
              this.faceObj.center[0] / 257 + this.faceObj.distance / 257,
            ],
          ],
          [0],
          [192, 192]
        );
        let face = await this.faceMeshModel.predict(cropImage.div(255));
        cropImage.dispose();
        const faceArrKeypoints = [];

        for (let i of SELECTED_FACE_POINTS) {
          faceArrKeypoints.push([
            face[1].dataSync()[i * 3],
            face[1].dataSync()[i * 3 + 1],
          ]);
        }
        this.faceObj.result = (
          await local2GlobalKeypoints(
            faceArrKeypoints,
            [
              this.faceObj.center,
              this.faceObj.distance * 2,
              this.faceObj.radians,
            ],
            192
          )
        ).reshape([24, 2]);
        face = this.faceObj.result.dataSync();

        poseStacks.push(poseResultKeypoints.slice());
        leftHandStacks.push(this.leftHand.result.arraySync());
        rightHandStacks.push(this.rightHand.result.arraySync());
        faceStacks.push(this.faceObj.result.arraySync());

        const toc = Date.now();
        timeMs = timeMs + (toc - tic);
        console.log(timeMs);
      }

      let poseStacksTensor = tf.tensor3d(poseStacks); // change array to tensor
      // tf.tensor3d(poseStacks).print()
      poseStacksTensor = poseStacksTensor.div(257).expandDims(0); // expand dim to [1, 32, 13, 2]
      // create noseStack from poseStack tensor
      let noseStack = poseStacksTensor.stridedSlice(
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1],
        11, // beginMask
        11, // endMask
        0, // ellipsisMask
        0, // newAxisMask
        4 // shrinkAxisMask
      );
      noseStack = noseStack.reshape([1, 16, 1, 2]);
      poseStacksTensor = poseStacksTensor.sub(
        tf.broadcastTo(noseStack, poseStacksTensor.shape)
      );
      // face
      let faceStacksTensor = tf
        .tensor3d(faceStacks)
        .div(257)
        .reshape([1, 16, 24, 2]);
      faceStacksTensor = faceStacksTensor.sub(
        tf.broadcastTo(noseStack, faceStacksTensor.shape)
      );
      // leftHand
      let leftHandStacksTensor = tf
        .tensor3d(leftHandStacks)
        .div(257)
        .reshape([1, 16, 21, 2]);
      leftHandStacksTensor = leftHandStacksTensor.sub(
        tf.broadcastTo(noseStack, leftHandStacksTensor.shape)
      );
      // rightHand
      let rightHandStacksTensor = tf
        .tensor3d(rightHandStacks)
        .div(257)
        .reshape([1, 16, 21, 2]);
      rightHandStacksTensor = rightHandStacksTensor.sub(
        tf.broadcastTo(noseStack, rightHandStacksTensor.shape)
      );
      const wordResult = await this.classifyModel.predict({
        pose_frames_input: poseStacksTensor,
        face_frames_input: faceStacksTensor,
        left_hand_frames_input: leftHandStacksTensor,
        right_hand_frames_input: rightHandStacksTensor,
      });
      // result to strings
      const predictionResult = wordResult[0].dataSync();

      poseStacksTensor.dispose();
      faceStacksTensor.dispose();
      leftHandStacksTensor.dispose();
      rightHandStacksTensor.dispose();

      this.faceObj.result.dispose();
      this.leftHand.result.dispose();
      this.rightHand.result.dispose();

      tf.engine().endScope();
      resolve({
        resultLabel:
          LABELS[predictionResult.indexOf(Math.max(...predictionResult))],
        resultArray: predictionResult,
      });
    });
  }

  async handPrediction(radians, distanceHand, center, imageInputTensor) {
    const input_tensor = imageInputTensor;

    const rotatedHandImage = tf.image.rotateWithOffset(
      input_tensor,
      radians,
      0,
      center
    );
    input_tensor.dispose();
    tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 100 * 1024 * 1024);
    const cropedHandImage = tf.image.cropAndResize(
      rotatedHandImage,
      [
        [
          center[1] - (distanceHand * 2) / 257,
          center[0] - (distanceHand * 2) / 257,
          center[1] + (distanceHand * 2) / 257,
          center[0] + (distanceHand * 2) / 257,
        ],
      ],
      [0],
      [256, 256]
    );

    // let handBuffer = cropedHandImage.pad([[0, 0], [0, 0], [0, 0], [0, 1]], 255).reshape([-1]).dataSync()
    let handBuffer = 0;
    const hand = await this.handPoseModel.predict(cropedHandImage.div(255));
    cropedHandImage.dispose();
    let handSelected = hand[2].dataSync();
    let handArrKeypoints = [];
    for (
      let i = 0, point = 0, n = handSelected.length;
      i < n;
      i += 3, point += 1
    ) {
      handArrKeypoints.push([handSelected[i], handSelected[i + 1]]);
    }
    let handResultKeypoints = await local2GlobalKeypoints(
      handArrKeypoints,
      [[center[0] * 257, center[1] * 257], distanceHand * 4, radians],
      256
    );
    // hand = hand[2].div(256).dataSync();
    let handKp = handResultKeypoints;
    return [handBuffer, handKp];
  }

  // predict pose hand face
  // return image and collec the result as stack to signing
  // predict classify

  async predictImage(image) {
    return new Promise(async (resolve) => {
      tf.engine().startScope();
      tf.env().set("WEBGL_PACK_DEPTHWISECONV", true);
      tf.env().set("WEBGL_FLUSH_THRESHOLD", 1);
      let timeMs = 0;
      const tic = Date.now();
      // console.log('image: ', image);
      const imageInputTensor = tf
        .slice(
          tf.tensor3d(Array.from(image.data), [257, 257, 4]),
          [0, 0, 0],
          [257, 257, 3]
        )
        .toFloat()
        .expandDims(0);
      const inputTensor = imageInputTensor.div(127.5).sub(1);
      // imageTensorForPredictionStack.push(inputTensor.dataSync()); // this for batch predictioon

      // if test batch, comment this --
      const poseResult = await this.poseModel?.predict({
        input: inputTensor,
      });
      inputTensor.dispose();

      // console.log(poseResult);
      const POSE = await decodePose(
        poseResult[0].squeeze(0),
        poseResult[1].squeeze(0),
        16
      );
      // console.log(POSE);
      // -- to this

      const poseResultKeypoints = [];
      for (let i = 0, n = POSE.keypoints.length; i < n; i += 1) {
        if (SELECTED_POSENET_JOINTS.includes(POSE.keypoints[i].part)) {
          // console.log(pose.keypoints[i].part, ' index : ', i)
          let THRESHOULD = 0.35;
          if (POSE.keypoints[i].score < THRESHOULD) {
            POSE.keypoints[i].position.x = 0;
            POSE.keypoints[i].position.y = 0;
          }
          poseResultKeypoints.push([
            POSE.keypoints[i].position.x,
            POSE.keypoints[i].position.y,
          ]);
          if (POSE.keypoints[i].part === "leftWrist") {
            if (POSE.keypoints[i].score < THRESHOULD) {
              this.leftHand.isFound = false;
            } else {
              this.leftHand.isFound = true;
              this.leftHand.writsKeypoint = [
                POSE.keypoints[i].position.x,
                POSE.keypoints[i].position.y,
              ];
            }
          }

          if (POSE.keypoints[i].part === "rightWrist") {
            if (POSE.keypoints[i].score < THRESHOULD) {
              this.rightHand.isFound = false;
            } else {
              this.rightHand.isFound = true;
              this.rightHand.writsKeypoint = [
                POSE.keypoints[i].position.x,
                POSE.keypoints[i].position.y,
              ];
            }
          }

          if (POSE.keypoints[i].part === "leftMidfin") {
            if (POSE.keypoints[i].score < THRESHOULD) {
              this.leftHand.isFound = false;
            } else {
              this.leftHand.isFound = true;
              this.leftHand.midfinKeypoint = [
                POSE.keypoints[i].position.x,
                POSE.keypoints[i].position.y,
              ];
            }
          }

          if (POSE.keypoints[i].part === "rightMidfin") {
            if (POSE.keypoints[i].score < THRESHOULD) {
              this.rightHand.isFound = false;
            } else {
              this.rightHand.isFound = true;
              this.rightHand.midfinKeypoint = [
                POSE.keypoints[i].position.x,
                POSE.keypoints[i].position.y,
              ];
            }
          }

          if (POSE.keypoints[i].part === "rightEar") {
            this.faceObj.rightEarKeypoint = [
              POSE.keypoints[i].position.x,
              POSE.keypoints[i].position.y,
            ];
          }
          if (POSE.keypoints[i].part === "leftEar") {
            this.faceObj.leftEarKeypoint = [
              POSE.keypoints[i].position.x,
              POSE.keypoints[i].position.y,
            ];
          }
        }
      }
      this.faceObj.radians = await getRadians(
        this.faceObj.leftEarKeypoint,
        this.faceObj.rightEarKeypoint,
        1
      );
      this.leftHand.radians = await getRadians(
        this.leftHand.writsKeypoint,
        this.leftHand.midfinKeypoint,
        2
      );
      this.rightHand.radians = await getRadians(
        this.rightHand.writsKeypoint,
        this.rightHand.midfinKeypoint,
        2
      );

      this.faceObj.center = [
        (this.faceObj.rightEarKeypoint[0] + this.faceObj.leftEarKeypoint[0]) /
          2,
        (this.faceObj.rightEarKeypoint[1] + this.faceObj.leftEarKeypoint[1]) /
          2,
      ];
      this.faceObj.distance = await getDistance(
        this.faceObj.leftEarKeypoint,
        this.faceObj.rightEarKeypoint
      );

      this.leftHand.center = this.leftHand.midfinKeypoint;
      this.leftHand.distance = await getDistance(
        this.leftHand.writsKeypoint,
        this.leftHand.midfinKeypoint
      );

      this.rightHand.center = this.rightHand.midfinKeypoint;
      this.rightHand.distance = await getDistance(
        this.rightHand.writsKeypoint,
        this.rightHand.midfinKeypoint
      );

      const leftHandResult = await this.handPrediction(
        this.leftHand.radians,
        this.leftHand.distance,
        [this.leftHand.center[0] / 257, this.leftHand.center[1] / 257],
        imageInputTensor
      );

      if (this.leftHand.isFound) {
        this.leftHand.result = leftHandResult[1];
      } else if (!this.leftHand.isFound) {
        this.leftHand.result = tf.zeros([21, 2]);
      }

      const rightHandResult = await this.handPrediction(
        this.rightHand.radians,
        this.rightHand.distance,
        [this.rightHand.center[0] / 257, this.rightHand.center[1] / 257],
        imageInputTensor
      );
      // rightHand.imageBuffer = rightHandResult[0]

      if (this.rightHand.isFound) {
        this.rightHand.result = rightHandResult[1];
      } else if (!this.rightHand.isFound) {
        this.rightHand.result = tf.zeros([21, 2]);
      }

      // const leftHandDebug = this.leftHand.result.dataSync();
      // const rightHandDebug = this.rightHand.result.dataSync();

      const input_tensor = imageInputTensor;
      const faceRotatedImage = tf.image.rotateWithOffset(
        input_tensor,
        this.faceObj.radians,
        0,
        [this.faceObj.center[0] / 257, this.faceObj.center[1] / 257]
      );
      input_tensor.dispose();
      tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 100 * 1024 * 1024);
      const cropImage = tf.image.cropAndResize(
        faceRotatedImage,
        [
          [
            this.faceObj.center[1] / 257 - this.faceObj.distance / 257,
            this.faceObj.center[0] / 257 - this.faceObj.distance / 257,
            this.faceObj.center[1] / 257 + this.faceObj.distance / 257,
            this.faceObj.center[0] / 257 + this.faceObj.distance / 257,
          ],
        ],
        [0],
        [192, 192]
      );
      let face = await this.faceMeshModel.predict(cropImage.div(255));
      cropImage.dispose();
      const faceArrKeypoints = [];

      for (let i of SELECTED_FACE_POINTS) {
        faceArrKeypoints.push([
          face[1].dataSync()[i * 3],
          face[1].dataSync()[i * 3 + 1],
        ]);
      }
      this.faceObj.result = (
        await local2GlobalKeypoints(
          faceArrKeypoints,
          [
            this.faceObj.center,
            this.faceObj.distance * 2,
            this.faceObj.radians,
          ],
          192
        )
      ).reshape([24, 2]);
      let pose = poseResultKeypoints.slice();
      face = this.faceObj.result.arraySync();
      let leftHand = this.leftHand.result.arraySync();
      let rightHand = this.rightHand.result.arraySync();

      this.faceObj.result.dispose();
      this.leftHand.result.dispose();
      this.rightHand.result.dispose();

      tf.engine().endScope();

      // console.log('leftHand: ', leftHand);
      // console.log('rightHand: ', rightHand);

      const toc = Date.now();
      timeMs = timeMs + (toc - tic);
      console.log(timeMs);
      resolve({
        pose: pose,
        face: face,
        leftHand: leftHand,
        rightHand: rightHand,
      });
    });
  }

  async predictSign(inputStack) {
    return new Promise(async (resolve) => {
      tf.engine().startScope();
      tf.env().set("WEBGL_PACK_DEPTHWISECONV", true);
      tf.env().set("WEBGL_FLUSH_THRESHOLD", 1);
      let poseStacksTensor = tf.tensor3d(inputStack.poseStack);
      poseStacksTensor = poseStacksTensor.div(257).expandDims(0); // expand dim to [1, 32, 13, 2]
      // create noseStack from poseStack tensor
      let noseStack = poseStacksTensor.stridedSlice(
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1],
        11, // beginMask
        11, // endMask
        0, // ellipsisMask
        0, // newAxisMask
        4 // shrinkAxisMask
      );
      noseStack = noseStack.reshape([1, 16, 1, 2]);
      poseStacksTensor = poseStacksTensor.sub(
        tf.broadcastTo(noseStack, poseStacksTensor.shape)
      );
      // face
      let faceStacksTensor = tf
        .tensor3d(inputStack.faceStack)
        .div(257)
        .reshape([1, 16, 24, 2]);
      faceStacksTensor = faceStacksTensor.sub(
        tf.broadcastTo(noseStack, faceStacksTensor.shape)
      );
      // leftHand
      let leftHandStacksTensor = tf
        .tensor3d(inputStack.leftHandStack)
        .div(257)
        .reshape([1, 16, 21, 2]);
      leftHandStacksTensor = leftHandStacksTensor.sub(
        tf.broadcastTo(noseStack, leftHandStacksTensor.shape)
      );
      // rightHand
      let rightHandStacksTensor = tf
        .tensor3d(inputStack.rightHandStack)
        .div(257)
        .reshape([1, 16, 21, 2]);
      rightHandStacksTensor = rightHandStacksTensor.sub(
        tf.broadcastTo(noseStack, rightHandStacksTensor.shape)
      );
      const wordResult = await this.classifyModel.predict({
        pose_frames_input: poseStacksTensor,
        face_frames_input: faceStacksTensor,
        left_hand_frames_input: leftHandStacksTensor,
        right_hand_frames_input: rightHandStacksTensor,
      });
      // result to strings
      const predictionResult = wordResult[0].dataSync();

      poseStacksTensor.dispose();
      faceStacksTensor.dispose();
      leftHandStacksTensor.dispose();
      rightHandStacksTensor.dispose();

      tf.engine().endScope();

      const ResultArray = LABELS.map((item, index) => {
        item = [item, predictionResult[index]];
        return item;
      });
      resolve({
        resultLabel:
          LABELS[predictionResult.indexOf(Math.max(...predictionResult))],
        resultArray: ResultArray,
      });
    });
  }
}
