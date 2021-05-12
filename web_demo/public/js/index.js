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

import { initVideoSeleciton } from "./selection.js";
import { setupCamera, captureImage } from "./record.js";

import SignLanguageClassifyModel from "./ML/signClassify.js";
import { drawResult } from "./drawkeypoints.js";
import { removeChild, checkArrayMatch, isMobile } from "./utils.js";
import initHelpmodal from "./helpmodal.js";

window.recoil = {
  selectSign: "",
  recordClickable: true,
  isModelReady: false,
  pageState: "intro",
};

$(document).ready(() => {
  // page state management
  const page_changeState = (input) => {
    window.recoil.pageState = input;
    const processingModal = document.querySelector(".processing-modal");
    const processingText = document.querySelector(".processing-text");
    const mobileModal = document.querySelector(".mobile-device-modal");
    const backgrounddiv = document.querySelector(".background");
    const mainSection = document.querySelector(".main-section");
    const introSection = document.querySelector(".intro-section");
    const demoSection = document.querySelector(".demo-section");
    const recordSection = document.querySelector(".record-section");
    const recordResult = document.querySelector(".record-result");

    switch (input) {
      case "mobilemodal":
        const allElement = document.querySelectorAll(
          "body > div:not(.mobile-device-modal)"
        );
        allElement.forEach((el) => (el.style.display = "none"));
        mobileModal.style.display = "flex";
        break;
      case "intro":
        mainSection.style.transform = "translateX(0)";
        processingModal.style.display = "none";
        introSection.style.opacity = 1;
        demoSection.style.opacity = 0;
        recordSection.style.opacity = 0;
        recordResult.style.opacity = 0;
        break;
      case "idle":
        // recordIdle.style.opacity = "1";
        // recordIdle.style.zIndex = "2";

        mainSection.style.transform = "translateX(calc(-1/3 * 100% - 3px))";
        introSection.style.opacity = 0;
        demoSection.style.opacity = 1;
        recordSection.style.opacity = 1;
        recordResult.style.opacity = 0;
        // recordResult.style.opacity = "0";
        // recordResult.style.zIndex = "1";

        processingModal.style.display = "none";
        console.log("idle");
        break;
      case "loadingmodel":
        processingModal.style.display = "flex";
        processingText.innerHTML = "Preparing model";
        console.log("loading model");
        break;
      case "processingmodel":
        processingModal.style.display = "flex";
        processingText.innerHTML = "Processing model";
        console.log("processing model");
        break;
      case "upload":
        // backgrounddiv.style.backgroundColor = "unset";
        backgrounddiv.style.zIndex = -1;

        processingModal.style.display = "flex";
        processingText.innerHTML = "upload data to cloud";
        console.log("uploading data model");
        break;
      case "result":
        // backgrounddiv.style.backgroundColor = "unset";
        backgrounddiv.style.zIndex = -1;

        // recordIdle.style.opacity = "0";
        // recordIdle.style.zIndex = "1";

        mainSection.style.transform = "translateX(calc(-2/3 * 100% - 3px))";
        introSection.style.opacity = 0;
        demoSection.style.opacity = 0;
        recordSection.style.opacity = 0;
        recordResult.style.opacity = 1;
        // recordResult.style.opacity = "1";
        // recordResult.style.zIndex = "2";
        processingModal.style.display = "none";
        console.log("result");
        break;
      case "result_no":
        backgrounddiv.style.backgroundColor = "rgb(80,80,80,0.7)";
        backgrounddiv.style.zIndex = 50;
      default:
        break;
    }
  };
  initHelpmodal();

  console.log("getting start ready!");
  initVideoSeleciton();

  const classifyModel = new SignLanguageClassifyModel();

  const initmodel = async () => {
    await classifyModel.initModel();
    window.recoil.isModelReady = true;
    // page_changeState("intro");
    console.log("init model finish");
    if (window.recoil.pageState === "processingmodel") page_changeState("idle");
  };
  function notMobile() {
    if (isMobile()) {
      page_changeState("mobilemodal");

      return false;
    }
    return true;
  }
  page_changeState("intro");
  notMobile() && initmodel();

  /**
   * Flow:
   * init video
   * init model
   * set upCamera
   * click record
   * count down to capture screen
   * capture image to stack
   * send image stack to ml
   * ml process
   * set state to result
   * show result [table, drawImage, sort accuracy sign result]
   */

  let showResultCanvas = 0;
  const sliderFrame = document.getElementById("frame-canvas-slider");
  const selectFrameResult = (index) => {
    const previousCanvas = document.getElementById(
      `image-frame-${showResultCanvas}`
    );
    if (previousCanvas) previousCanvas.style.display = "none";
    const selectedCanvas = document.getElementById(`image-frame-${index}`);
    if (selectedCanvas) selectedCanvas.style.display = "flex";
    const frameCanvasText = document.getElementById("frame-canvas-text-id");
    frameCanvasText.innerHTML = `frame: ${Number(index) + 1}`;
    sliderFrame.value = index;
    showResultCanvas = index;
  };
  const updateVisiblePart = (index) => {
    const visiblePart = FRAME_KEYPOINTS_TABLE[+index];

    const poseEl = document.getElementById("visible-part-pose"),
      leftHandEl = document.getElementById("visible-part-left-hand"),
      rightHandEl = document.getElementById("visible-part-right-hand"),
      faceEl = document.getElementById("visible-part-face");

    const addOrRemove = (visible) => (visible ? "add" : "remove");

    poseEl.classList[addOrRemove(visiblePart.pose)]("active");
    leftHandEl.classList[addOrRemove(visiblePart.leftHand)]("active");
    rightHandEl.classList[addOrRemove(visiblePart.rightHand)]("active");
    faceEl.classList[addOrRemove(visiblePart.face)]("active");
  };
  sliderFrame.oninput = function (e) {
    selectFrameResult(this.value);
    updateVisiblePart(this.value);

    const progress = +e.target.value / (+e.target.max - +e.target.min);

    e.target.style.background = `linear-gradient(to right,
      #1a73e8 0%,
      #1a73e8 ${progress * 100}%,
      #ffffff ${progress * 100}%,
      #ffffff 100%
      )`;
  };

  const IMAGE_STACK = [];
  const PREDICTION_IMAGE_STACK = [];
  const FRAME_KEYPOINTS_TABLE = [];
  const RESULT_POSE_STACK = [];
  const RESULT_FACE_STACK = [];
  const RESULT_LEFTHAND_STACK = [];
  const RESULT_RIGHTHAND_STACK = [];
  const topFiveResultArr = [];
  let signingResult = "";

  const startClassify = () => {
    const thres = (IMAGE_STACK.length - 5) / 16;
    const imageTime = [];
    for (let i = 0; i < 16; i++) {
      imageTime.push(Math.round(thres * i));
    }
    console.log("image time stack: ", imageTime);
    for (const time of imageTime) {
      PREDICTION_IMAGE_STACK.push(IMAGE_STACK[time + 3]);
    }
    console.log("prediciton image stack: ", PREDICTION_IMAGE_STACK);
    IMAGE_STACK.length = 0; // clear image stack
    const predictionImage = async () => {
      for (const image_index in PREDICTION_IMAGE_STACK) {
        console.log(image_index);
        const result = await classifyModel.predictImage(
          PREDICTION_IMAGE_STACK[image_index].imageData
        );
        console.log(result);
        const isPose = !checkArrayMatch(result.pose, [0, 0]);
        const isFace = !checkArrayMatch(result.face, [0, 0]);
        const isLeftHand = !checkArrayMatch(result.leftHand, [0, 0]);
        const isRightHand = !checkArrayMatch(result.rightHand, [0, 0]);
        FRAME_KEYPOINTS_TABLE.push({
          frame: image_index,
          pose: isPose,
          face: isFace,
          leftHand: isLeftHand,
          rightHand: isRightHand,
        });
        RESULT_POSE_STACK.push(result.pose);
        RESULT_FACE_STACK.push(result.face);
        RESULT_LEFTHAND_STACK.push(result.leftHand);
        RESULT_RIGHTHAND_STACK.push(result.rightHand);
      }

      console.log("finished predict image: pose, face, hand");

      console.log("update visible part after finish prediction");
      updateVisiblePart(sliderFrame.value);
      const resultStack = {
        poseStack: RESULT_POSE_STACK,
        faceStack: RESULT_FACE_STACK,
        leftHandStack: RESULT_LEFTHAND_STACK,
        rightHandStack: RESULT_RIGHTHAND_STACK,
      };
      console.log("result stack: ", resultStack);
      const classifyResult = await classifyModel.predictSign(resultStack);
      const cmp = (a, b) => {
        return b[1] > a[1] ? 1 : -1;
      };
      // sorted the rank of sign result
      const sortedArray = classifyResult.resultArray.sort(cmp);
      console.log(sortedArray);

      // get all stack keypoints and image stack send to draw key points
      const canvasWrapperEl = document.getElementById(
        "frame-canvas-wrapper-id"
      );
      for (const i in PREDICTION_IMAGE_STACK) {
        const canvasEl = drawResult({
          imageData: PREDICTION_IMAGE_STACK[i].imageData,
          resultKeypoints: {
            poseStack: RESULT_POSE_STACK[i],
            faceStack: RESULT_FACE_STACK[i],
            leftHandStack: RESULT_LEFTHAND_STACK[i],
            rightHandStack: RESULT_RIGHTHAND_STACK[i],
          },
        });
        if (canvasEl !== null && canvasWrapperEl !== null) {
          canvasEl.classList.add("result-kp-image");
          canvasEl.id = `image-frame-${i}`;
          canvasEl.style.display = i == showResultCanvas ? "flex" : "none";
          canvasWrapperEl.appendChild(canvasEl);
        }
      }

      // update signingResult
      signingResult = classifyResult.resultLabel;

      // update keypoint to table analyst
      const frameParentTable = document.getElementById("frame-table-body-id");
      await Promise.all(
        FRAME_KEYPOINTS_TABLE.map((item, index) => {
          // append data to html table
          const thisTable = document.createElement("tr");
          thisTable.setAttribute("key", index);
          thisTable.addEventListener("click", () => {
            selectFrameResult(index);
          });
          const frameNode = document.createElement("td");
          frameNode.innerHTML = Number(item.frame) + 1;
          const isPoseNode = document.createElement("td");
          isPoseNode.innerHTML = item.pose ? "yes" : "no";
          isPoseNode.style.backgroundColor = item.pose ? "#7ecbbd" : "#de5246";
          const isFaceNode = document.createElement("td");
          isFaceNode.innerHTML = item.face ? "yes" : "no";
          isFaceNode.style.backgroundColor = item.face ? "#7ecbbd" : "#de5246";
          const isLeftHandNode = document.createElement("td");
          isLeftHandNode.innerHTML = item.leftHand ? "yes" : "no";
          isLeftHandNode.style.backgroundColor = item.leftHand
            ? "#7ecbbd"
            : "#de5246";
          const isRightHandNode = document.createElement("td");
          isRightHandNode.innerHTML = item.rightHand ? "yes" : "no";
          isRightHandNode.style.backgroundColor = item.rightHand
            ? "#7ecbbd"
            : "#de5246";

          thisTable.appendChild(frameNode);
          thisTable.appendChild(isPoseNode);
          thisTable.appendChild(isFaceNode);
          thisTable.appendChild(isLeftHandNode);
          thisTable.appendChild(isRightHandNode);
          frameParentTable.appendChild(thisTable);
        })
      );

      // update to result state
      // update top 5 result
      // remove exits table
      const parentTable = document.getElementById("table-body");
      console.log("check remove child");
      for (let i = 0; i < 5; i++) {
        topFiveResultArr.push({
          sign: sortedArray[i][0],
          acc: (sortedArray[i][1] * 100).toFixed(2),
        });

        // create top 5 table
        const thisTable = document.createElement("tr");
        thisTable.setAttribute("key", i);
        const rankNode = document.createElement("td");
        rankNode.innerHTML = i + 1;
        const resultNode = document.createElement("td");
        resultNode.innerHTML = sortedArray[i][0];
        const accNode = document.createElement("td");
        accNode.innerHTML = sortedArray[i][1].toFixed(2);
        thisTable.appendChild(rankNode);
        thisTable.appendChild(resultNode);
        thisTable.appendChild(accNode);
        console.log("append table child");
        parentTable.appendChild(thisTable);
      }
      page_changeState("result");
    };
    predictionImage();
  };

  const handleCapture = () => {
    const time_startCapture = +new Date();
    const captureFrame = () => {
      const imageCaptured = captureImage();
      if (imageCaptured) IMAGE_STACK.push(imageCaptured);
      const time_now = +new Date();
      if (time_now - time_startCapture > 3000) {
        console.log("finish capture image");
        // record_changeState("idle");
        page_changeState("processingmodel");
        startClassify();
        return;
      } else {
        requestAnimationFrame(captureFrame);
      }
    };
    requestAnimationFrame(captureFrame);
  };

  $("#intro-next-btn").on("click", () => {
    if (window.recoil.isModelReady) {
      page_changeState("idle");
    } else {
      page_changeState("processingmodel");
    }
    setupCamera();
  });

  $("#record-btn-id").on("click", () => {
    // count down 3 sec
    if (window.recoil.recordClickable) {
      window.recoil.recordClickable = false;
      const countdownElem = document.getElementById("countdown-text");
      let count = 0;
      countdownElem.innerHTML = 3;
      const CountDownInterval = setInterval(() => {
        countdownElem.innerHTML = 2 - count;
        count += 1;
        console.log(count);
        if (count === 3) {
          countdownElem.innerHTML = "";
          // record_changeState("record");
          clearInterval(CountDownInterval);
          handleCapture();
        }
      }, 1000);
    }
  });

  $("#language-hksl-btn").on("click", (e) => {
    const jsl_table = document.querySelector(".jsl-sign-table");
    const hksl_table = document.querySelector(".hksl-sign-table");
    jsl_table.style.display = "none";
    hksl_table.style.display = "block";

    document.getElementsByClassName("language-btn").forEach((item) => {
      item.classList.remove("active");
    });
    e.target.classList.add("active");
  });
  $("#language-jsl-btn").on("click", (e) => {
    const jsl_table = document.querySelector(".jsl-sign-table");
    const hksl_table = document.querySelector(".hksl-sign-table");
    jsl_table.style.display = "block";
    hksl_table.style.display = "none";

    document.getElementsByClassName("language-btn").forEach((item) => {
      item.classList.remove("active");
    });
    e.target.classList.add("active");
  });

  // click tryagain
  const clearStack = () => {
    removeChild("#frame-table-body-id");
    removeChild("#table-body");
    removeChild("#frame-canvas-wrapper-id");
    IMAGE_STACK.length = 0;
    PREDICTION_IMAGE_STACK.length = 0;
    FRAME_KEYPOINTS_TABLE.length = 0;
    RESULT_POSE_STACK.length = 0;
    RESULT_FACE_STACK.length = 0;
    RESULT_LEFTHAND_STACK.length = 0;
    RESULT_RIGHTHAND_STACK.length = 0;
    topFiveResultArr.length = 0;
  };
  $("#try-again-btn").on("click", () => {
    clearStack();
    window.recoil.recordClickable = true;
    page_changeState("idle");
  });
});
