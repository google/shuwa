<h1 align='center'> Shuwa </h1>
This is the example of applying ML model that coverted from python. We using simple of html/css/js implementation which is easy to read and hacking.

our **sign-language videos demo** were signed and recorded by [The Nippon Foundation (Japan)](https://www.nippon-foundation.or.jp/en), [Chinese University of Hong Kong (CUHK)](https://www.cuhk.edu.hk/), and deaf advocacy groups across APAC.

# getting start

## Prerequisites

node.js for hosting the pre-set server on `server.js` and javascript package management: `yarn` or `npm` either preferred.

## Installation and Start the server

- clone the repo

```
git clone https://github.com/bitstudio/shuwa.git
```

- go to web_demo directory `cd web_demo`
- install the server packages `yarn install` or `npm install`
- starting the server `yarn start` or `npm start`\
  runs the app in the development mode. open http://localhost:8000 to view it in the browser.

# structure

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
initilize model
setup camera
recording
send captured images to ML
ML process
show result
back to record page
```

# Code implementation

## **signClassify**

there is a script that was implemented for classify the sign-language model name `signClassify.js`, located at `./js/ML` directory.

### usage example

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

there are 4 methods but they are aim the same purpose which is initialize the tensorflow model. `initModel()` was use for initialize all models in one call: pose, face, hand, and classify in one call. The others, `initPose()`, `initFace()`, `initHand()`, and `initClassify()`, were created in-case if you want to optimize the model loading.

#### **predict**

`predict(imagestack)`\
this methods runs all prediction in once, which are detect pose face hand, and then classify. `imagestack` input argument must have the lenght of list equal to 16 (16 frames)

```js
const classifyResult = await classifyModel.predict([...`16 images`]);
```

We also develop the functions `predictImage(image)` to predict each image just in case that you want to use _keypoints_ to do something, such as draw the detected line. Then, collect the keypoints result in 16 stacks of array in the objects `resStack`. Send it to `predictSign(resStack)` to receive classifyResult data.

```js
const resStack = {
  poseStack: [..."pose keypoints"],
  faceStack: [..."face keypoints"],
  leftHandStack: [..."left-hand keypoints"],
  rightHandStack: [..."right-hand keypoints"],
};

const classifyResult = await classifyModel.predictSign(resStack);
```
