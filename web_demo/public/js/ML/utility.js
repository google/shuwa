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

import { partNames } from "./const.js";

const NUM_KEYPOINTS = partNames.length;

function mod(a, b) {
  // a: tf.Tensor1D, b: number
  return tf.tidy(() => {
    const floored = a.div(tf.scalar(b, "int32"));

    return a.sub(floored.mul(tf.scalar(b, "int32")));
  });
}

function argmax2d(inputs) {
  // inputs: tf.Tensor3D
  const [height, width, depth] = inputs.shape;

  return tf.tidy(() => {
    const reshaped = inputs.reshape([height * width, depth]);
    const coords = reshaped.argMax(0);

    const yCoords = coords.div(tf.scalar(width, "int32")).expandDims(1);
    const xCoords = mod(coords, width).expandDims(1); // coords: tf.Tensor1D

    return tf.concat([yCoords, xCoords], 1);
  }); // tf.Tensor2D
}

function getOffsetPoint(y, x, keypoint, offsetsBuffer) {
  return {
    y: offsetsBuffer.get(y, x, keypoint),
    x: offsetsBuffer.get(y, x, keypoint + NUM_KEYPOINTS),
  };
}

function getOffsetVectors(heatMapCoordsBuffer, offsetsBuffer) {
  const result = [];

  for (let keypoint = 0; keypoint < NUM_KEYPOINTS; keypoint++) {
    const heatmapY = heatMapCoordsBuffer.get(keypoint, 0).valueOf();
    const heatmapX = heatMapCoordsBuffer.get(keypoint, 1).valueOf();

    const { x, y } = getOffsetPoint(
      heatmapY,
      heatmapX,
      keypoint,
      offsetsBuffer
    );

    result.push(y);
    result.push(x);
  }

  return tf.tensor2d(result, [NUM_KEYPOINTS, 2]);
}

function getOffsetPoints(heatMapCoordsBuffer, outputStride, offsetsBuffer) {
  return tf.tidy(() => {
    const offsetVectors = getOffsetVectors(heatMapCoordsBuffer, offsetsBuffer);

    return heatMapCoordsBuffer
      .toTensor()
      .mul(tf.scalar(outputStride, "int32"))
      .toFloat()
      .add(offsetVectors);
  });
}

function getPointsConfidence(heatmapScores, heatMapCoords) {
  const numKeypoints = heatMapCoords.shape[0];
  const result = new Float32Array(numKeypoints);

  for (let keypoint = 0; keypoint < numKeypoints; keypoint++) {
    const y = heatMapCoords.get(keypoint, 0);
    const x = heatMapCoords.get(keypoint, 1);
    result[keypoint] = heatmapScores.get(y, x, keypoint);
  }

  return result;
}

export const decodePose = async (heatmapScores, offsets, outputStride) => {
  let totalScore = 0.0;
  const heatmapValues = argmax2d(heatmapScores);
  const allTensorBuffers = await Promise.all([
    heatmapScores.buffer(),
    offsets.buffer(),
    heatmapValues.buffer(),
  ]);

  const scoresBuffer = allTensorBuffers[0];
  const offsetsBuffer = allTensorBuffers[1];
  const heatmapValuesBuffer = allTensorBuffers[2];

  const offsetPoints = getOffsetPoints(
    heatmapValuesBuffer,
    outputStride,
    offsetsBuffer
  );
  const offsetPointsBuffer = await offsetPoints.buffer();

  const keypointConfidence = Array.from(
    getPointsConfidence(scoresBuffer, heatmapValuesBuffer)
  );

  const keypoints = keypointConfidence.map((score, keypointId) => {
    totalScore += score;
    return {
      position: {
        y: offsetPointsBuffer.get(keypointId, 0),
        x: offsetPointsBuffer.get(keypointId, 1),
      },
      part: partNames[keypointId],
      score,
    };
  });

  heatmapValues.dispose();
  offsetPoints.dispose();

  return { keypoints, score: totalScore / keypoints.length };
};

export const local2GlobalKeypoints = (
  localKeypoints,
  globalBox,
  netSize = 256
) => {
  return new Promise((resolve, reject) => {
    const globalBoxCenter = globalBox[0];
    const globalBoxSize = globalBox[1];
    const globalBoxRotation = globalBox[2];

    try {
      const localKeypointsTensor = tf.tensor2d(localKeypoints);
      // denormalize keypoints.
      const localKeypointsTensorMul = localKeypointsTensor.mul(
        globalBoxSize / netSize
      );
      // center at 0.
      const localKeypointsNorm = localKeypointsTensorMul.sub(globalBoxSize / 2);
      // rotate keypoints.

      const rotMat = buildRotationMatrix(-globalBoxRotation);
      const localKeypointsNormMul = tf.matMul(localKeypointsNorm, rotMat);

      resolve(localKeypointsNormMul.add(globalBoxCenter));
    } catch (err) {
      reject(err);
    }
  });
};

export function buildRotationMatrix(rotation) {
  const cosA = Math.cos(rotation);
  const sinA = Math.sin(rotation);
  return [
    [cosA, -sinA],
    [sinA, cosA],
  ];
}

export function getRadians(pointA, pointB, rotationFactor = 1) {
  return new Promise((resolve) => {
    const radians =
      Math.PI / rotationFactor -
      Math.atan2(-(pointB[1] - pointA[1]), pointB[0] - pointA[0]);
    resolve(
      radians - 2 * Math.PI * Math.floor((radians + Math.PI) / (2 * Math.PI))
    );
  });
}

export function getDistance(pointA, pointB) {
  return new Promise((resolve) => {
    resolve(
      Math.sqrt(
        Math.pow(pointA[0] - pointB[0], 2) + Math.pow(pointA[1] - pointB[1], 2)
      )
    );
  });
}
