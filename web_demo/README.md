# Shuwa Gesture Toolkit Web Demo

This is an example of using the models trained using the [Shuwa Gesture Toolkit](https://github.com/google/shuwa) and deploying it for a web application using Tensorflow.js

Our sign language sample videos were signed and recorded by [The Nippon Foundation (Japan)](https://www.nippon-foundation.or.jp/en), [Chinese University of Hong Kong (CUHK)](https://www.cuhk.edu.hk/), and deaf advocacy groups across APAC.

# Getting Started

## Prerequisites

node.js for hosting the pre-set server on `server.js` and javascript package management: `yarn` or `npm` either preferred.

## Installing and Running the Demo

- clone the repo

```
git clone https://github.com/google/shuwa.git
```

- Go to web_demo directory `cd web_demo`
- Install the server packages `yarn install` or `npm install`
- Start the server `yarn start` or `npm start`

# Structure

```c++
public
  index.html
  > model
  > js
    > ML // ML class implemented for using sign-lang Model
  > css
  > assets
    > icons
    > videos
```

# Project Flow

```
initialize model
setup camera
recording
send captured images to ML
ML process
show result
back to record page
```

# Usage Example

```js
import SignLanguageClassifyModel from "./ML/signClassify.js";

const classifyModel = new SignLanguageClassifyModel();

const initModel = async () => {
  await classifyModel.initModel();
  console.log("finish initialize model");
};
initModel();

/**
 * capture the 16 frames from web cam
 */
const images = [..."16 frames from web cam captured"];

const resStack = {
  poseStack: [],
  faceStack: [],
  leftHandStack: [],
  rightHandStack: [],
};
for (const image of images) {
  classifyModel.predictImage(image).then((res) => {
    console.log(res);
    /**
     * get the result keypoints of pose, face, lefthand, and righthand
     * you can draw them on the image using drawkeypoints.js
     */
    resStack.poseStack.push(res.pose);
    resStack.faceStack.push(res.face);
    resStack.leftHandStack.push(res.leftHand);
    resStack.rightHandStack.push(res.rightHand);
  });
}

const classifyResult = await classifyModel.predictSign(resStack);
console.log(classifyResult.resultLabel);
// classifyResult consist of 2 items: resutlLabel and resultArray
```

### Methods

#### **init**

`initModel()` initalizes all models needed in one call: pose, face, hand, and classify in one call.

The others, `initPose()`, `initFace()`, `initHand()`, and `initClassify()`, were created in-case if you want to optimize the model loading and load them independently.

#### **predict**

`predict(imagestack)`\
Runs the entire prediction pipeline, detecting pose, face, and hand, and then classifies the gesture.
`imagestack` input argument must have the length of list equal to 16 (16 frames)

```js
const classifyResult = await classifyModel.predict([...`16 images`]);
```

`predictImage(image)` predicts the keypoints for a single image, just in case you want to use _keypoints_ to do something, such as rendering the keypoints for a single frame. 
You can collect the keypoints result in 16 stacks of array in the objects `resStack` and send it to `predictSign(resStack)` to receive classifyResult data.

```js
const resStack = {
  poseStack: [..."pose keypoints"],
  faceStack: [..."face keypoints"],
  leftHandStack: [..."left-hand keypoints"],
  rightHandStack: [..."right-hand keypoints"],
};

const classifyResult = await classifyModel.predictSign(resStack);
```
