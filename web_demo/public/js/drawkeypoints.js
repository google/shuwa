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

"use strict";
export const drawResult = (input) => {
  console.log("draw keypoints");
  /**
   * canvas
   * create canvas
   * draw image
   *
   * pose:
   * draw keypoints
   * draw line
   *
   * face:
   * draw keypoints
   * draw line
   *
   * leftHand:
   * draw keypoints
   * draw line
   *
   * rightHand:
   * draw keypoints
   * draw line
   *
   * return canvas
   */
  const canvasEl = document.createElement("canvas");
  canvasEl.width = 257;
  canvasEl.height = 257;
  const ctx = canvasEl.getContext("2d");
  if (!ctx) return null;
  ctx.putImageData(input.imageData, 0, 0);

  // pose
  const posePoints = input.resultKeypoints.poseStack;
  ctx.beginPath();
  // check start, check [0,0]
  const posePath = [11, 9, 7, 5, 6, 8, 10, 12];
  let isMove = true;
  for (const poseKey of posePath) {
    // check value
    if (JSON.stringify(posePoints[poseKey]) === "[0,0]") {
      isMove = true;
    } else {
      // check is move
      if (isMove) {
        ctx.moveTo(posePoints[poseKey][0], posePoints[poseKey][1]);
        isMove = false;
      } else {
        // draw line
        ctx.lineTo(posePoints[poseKey][0], posePoints[poseKey][1]);
      }
    }
  }
  // ctx.moveTo(posePoints[11][0], posePoints[11][1]);
  // ctx.lineTo(posePoints[9][0], posePoints[9][1]);
  // ctx.lineTo(posePoints[7][0], posePoints[7][1]);
  // ctx.lineTo(posePoints[5][0], posePoints[5][1]);
  // ctx.lineTo(posePoints[6][0], posePoints[6][1]);
  // ctx.lineTo(posePoints[8][0], posePoints[8][1]);
  // ctx.lineTo(posePoints[10][0], posePoints[10][1]);
  // ctx.lineTo(posePoints[12][0], posePoints[12][1]);
  ctx.lineWidth = 3;
  ctx.strokeStyle = "cyan";
  ctx.stroke();

  // face
  ctx.fillStyle = "pink";
  for (const point of input.resultKeypoints.faceStack) {
    ctx.fillRect(point[0], point[1], 2, 2);
  }

  // leftHand
  ctx.fillStyle = "white";
  for (const point of input.resultKeypoints.leftHandStack) {
    ctx.fillRect(point[0], point[1], 2, 2);
  }

  // rightHand
  for (const point of input.resultKeypoints.rightHandStack) {
    ctx.fillRect(point[0], point[1], 2, 2);
  }

  return canvasEl;
};

/**
 * pose line path
 * 5 - 7 - 9 - 11
 * |
 * 6 - 8 - 10 - 12
 */
