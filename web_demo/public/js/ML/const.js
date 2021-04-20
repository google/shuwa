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

export const partNames = [
  "nose",
  "leftEye",
  "rightEye",
  "leftEar",
  "rightEar",
  "leftShoulder",
  "rightShoulder",
  "leftElbow",
  "rightElbow",
  "leftWrist", // 9
  "rightWrist", // 10
  "leftHip",
  "rightHip",
  "leftKnee",
  "rightKnee",
  "leftAnkle",
  "rightAnkle",
  "leftMidfin", // 17
  "rightMidfin", // 18
  /*
    left hand oreintation using 9-17
    right hand oreintation using 10-18
    */
];

export const SELECTED_POSENET_JOINTS = [
  "nose",
  "leftEye",
  "rightEye",
  "leftEar",
  "rightEar",
  "leftShoulder",
  "rightShoulder",
  "leftElbow",
  "rightElbow",
  "leftWrist",
  "rightWrist",
  "leftMidfin",
  "rightMidfin",
];

export const SELECTED_FACE_POINTS = [
  78,
  191,
  80,
  13,
  310,
  415,
  308,
  324,
  318,
  14,
  88,
  95,
  107,
  69,
  105,
  52,
  159,
  145,
  336,
  299,
  334,
  282,
  386,
  374,
];

export const LABELS = [
  "0_Idle",
  "Hksl_able_to",
  "Hksl_panda",
  "Hksl_bicycle-Jsl_bicycle",
  "Hksl_busy",
  "Hksl_carrot",
  "Hksl_chef",
  "Hksl_coffee",
  "Hksl_correct",
  "Hksl_dog",
  "Hksl_dragonfly",
  "Hksl_finish",
  "Hksl_flip_flops",
  "Hksl_fortune_teller",
  "Hksl_garden",
  "Hksl_ginger",
  "Hksl_good",
  "Hksl_grass",
  "Hksl_grey",
  "Hksl_hat-Jsl_cap", //19
  "Hksl_hearing_aid-Jsl_hearing_aid", //20
  "Hksl_high_heels",
  "Hksl_how_are_you",
  "Hksl_leisurely",
  "Hksl_let_it_go",
  "Hksl_lipstick",
  "Hksl_machine",
  "Hksl_matter",
  "Hksl_milk_tea",
  "Hksl_motorcycle-Jsl_motorcycle", //29
  "Hksl_obstruct",
  "Hksl_post_office",
  "Hksl_purple",
  "Hksl_read",
  "Hksl_shark",
  "Hksl_signature",
  "Hksl_soft_drink",
  "Hksl_special",
  "Hksl_strange",
  "Hksl_summer",
  "Hksl_sunglasses",
  "Hksl_supermarket",
  "Hksl_tiger",
  "Hksl_tomato",
  "Hksl_trouble",
  "Hksl_understand",
  "Hksl_watermelon",
  "Hksl_welcome",
  "Hksl_wine",
  "Hksl_winter-Jsl_winter", //49
  "Hksl_worry",
  "Hksl_write",
  "Hksl_yes",
  "Jsl_age",
  "Jsl_ahh",
  "Jsl_be_good_at",
  "Jsl_bird",
  "Jsl_brown",
  "Jsl_canteen",
  "Jsl_carrot",
  "Jsl_cat",
  "Jsl_department_store",
  "Jsl_dog",
  "Jsl_dragonfly",
  "Jsl_draught_beer",
  "Jsl_dream",
  "Jsl_earring",
  "Jsl_elevator",
  "Jsl_example",
  "Jsl_fly",
  "Jsl_geta",
  "Jsl_get_up",
  "Jsl_goldfish",
  "Jsl_gray",
  "Jsl_hello_with_one_hand",
  "Jsl_hello_with_two_hands",
  "Jsl_high_heels",
  "Jsl_hot_spring",
  "Jsl_illness",
  "Jsl_imagination",
  "Jsl_insect",
  "Jsl_japanese_radish",
  "Jsl_law",
  "Jsl_lemon",
  "Jsl_lie",
  "Jsl_milk_tea",
  "Jsl_orange_juice",
  "Jsl_oversleep",
  "Jsl_pine",
  "Jsl_programmer",
  "Jsl_rice_field",
  "Jsl_sports_athelets",
  "Jsl_strawberry",
  "Jsl_summer",
  "Jsl_swim",
  "Jsl_tell_say",
  "Jsl_unique_not_usual",
  "Jsl_watch",
  "Jsl_wine",
  "Jsl_yellow",
];

export const MULTI_TAGS = {
  Hksl_bicycle: 3,
  Jsl_bicycle: 3,
  Hksl_hat: 19,
  Jsl_cap: 19,
  Hksl_hearing_aid: 20,
  Jsl_hearing_aid: 20,
  Hksl_motorcycle: 29,
  Jsl_motorcycle: 29,
  Hksl_winter: 49,
  Jsl_winter: 49,
};
