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

export const removeChild = (inputClass) => {
  return new Promise((resolve) => {
    const parent = document.querySelector(inputClass);

    let child = parent.lastElementChild;
    while (child) {
      console.log("child: ", child);
      parent.removeChild(child);
      child = parent.lastElementChild;
    }
    resolve("finished");
  });
};

export const checkArrayMatch = (a, b) => {
  const z = a.map((item) => {
    return JSON.stringify(item);
  });
  return z.includes(JSON.stringify(b));
};

export const isMobile = () => {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
};
