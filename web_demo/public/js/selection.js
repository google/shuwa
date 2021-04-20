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

const formatSign = (sign) => {
  const splited = sign.split("_");
  splited.shift();
  return splited.join(" ");
};

export const initVideoSeleciton = async () => {
  /**
   * table
   * fetch available data
   * selection sign buffer
   * map available data to table
   */

  const label_list = await $.getJSON("./assets/label_list.json");
  console.log(label_list);

  const handleSelectSign = (group, input) => {
    window.recoil["selectSign"] = input;
    let pickerGroup = document.getElementsByClassName(group);

    for (var i = 0; i < pickerGroup.length; i++) {
      pickerGroup[i].classList.remove("active");
    }
    let picker = document.getElementById(input);
    console.log("picker: ", picker);
    picker.classList.add("active");

    const signLanguage = input.split("_")[0];
    const videoSrc = "./assets/videos/" + signLanguage + "/" + input + ".mp4";

    const outputVideo = document.getElementById("demo-video");
    outputVideo.src = videoSrc;
  };

  // add jsl selection list
  const jsl_table = document.createElement("div");
  jsl_table.classList.add("jsl-sign-table");
  jsl_table.style.display = "none";
  $(".demo-section-sign-wrapper-sign-table-wrapper").append(jsl_table);

  for (const jsl_sign of label_list.JSL_LABELS) {
    console.log(jsl_sign);
    const jsl_button = document.createElement("button");
    jsl_button.classList.add("jsl-sign-button");
    jsl_button.setAttribute("id", jsl_sign);
    jsl_button.innerHTML = formatSign(jsl_sign);

    jsl_button.addEventListener("click", () => {
      console.log("click: ", jsl_sign);
      handleSelectSign("jsl-sign-button", jsl_sign);
    });
    $(".jsl-sign-table").append(jsl_button);
  }

  // add hksl selection list
  const hksl_table = document.createElement("div");
  hksl_table.classList.add("hksl-sign-table");
  $(".demo-section-sign-wrapper-sign-table-wrapper").append(hksl_table);

  for (const hksl_sign of label_list.HKSL_LABELS) {
    const hksl_button = document.createElement("button");
    hksl_button.classList.add("hksl-sign-button");
    hksl_button.setAttribute("id", hksl_sign);
    hksl_button.innerHTML = formatSign(hksl_sign);

    hksl_button.addEventListener("click", () => {
      console.log("click: ", hksl_sign);
      handleSelectSign("hksl-sign-button", hksl_sign);
    });
    $(".hksl-sign-table").append(hksl_button);
  }
};
