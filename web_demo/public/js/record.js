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

"use-strict";

let cameraState = "idle";
export const setupCamera = () => {
  /**
   * init camera
   * init record button
   * handle record
   * update state record finish (3 secs)
   */

  let isCameraSetup = false;
  const videoOutput = document.getElementById("video-camera-id");
  const onLoadedVideo = () => {
    console.log("loadedVideoData");
    if (videoOutput !== null && isCameraSetup) {
      videoOutput.play();
    }
  };
  const setupCamera = async () => {
    const constraints = {
      audio: false,
      video: {
        facingMode: "user", // 'user' or 'environment'
      },
    };
    if (videoOutput !== null) {
      const mediaStream = await navigator.mediaDevices
        .getUserMedia(constraints)
        .catch((err) => {
          console.log(err.name);
          props.action();
          if (err.name === "NotAllowedError") {
            console.log("camera permission deniend");
            return;
          } else {
            console.log("camera undefined");
            return;
          }
        });
      if (mediaStream) {
        videoOutput.srcObject = mediaStream;
        isCameraSetup = true;
        onLoadedVideo();
        console.log(`--- set up camera ---`);
      }
    } else return;
  };
  setupCamera();
};

export const captureImage = () => {
  cameraState = "capturing";
  const canvasElem = document.getElementById("canvas-capture-id");
  const videoElem = document.getElementById("video-camera-id");

  const canvasCtx = canvasElem.getContext("2d");

  canvasCtx.clearRect(0, 0, canvasElem.width, canvasElem.height);

  canvasCtx?.drawImage(
    videoElem,
    (videoElem.videoWidth - videoElem.videoHeight) / 2,
    0,
    videoElem?.videoHeight,
    videoElem?.videoHeight,
    0,
    0,
    canvasElem?.width,
    canvasElem?.height
  );

  return {
    imageData: canvasCtx?.getImageData(
      0,
      0,
      canvasElem.width,
      canvasElem.height
    ),
    dataUrl: canvasElem?.toDataURL("image/png"),
  };
};
