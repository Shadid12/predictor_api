const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');

if (!cv.xmodules.face) {
  throw new Error('exiting: opencv4nodejs compiled without face module');
}

const basePath = './data/face-recognition';
const imgsPath = path.resolve(basePath, 'imgs');
const nameMappings = ['daryl', 'rick', 'negan'];

const imgFiles = fs.readdirSync(imgsPath);
 
const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);



const getFaceImage = (grayImg) => {
    const faceRects = classifier.detectMultiScale(grayImg).objects;
    if (!faceRects.length) {
      throw new Error('failed to detect faces');
    }
    return grayImg.getRegion(faceRects[0]);
};


const images = imgFiles
    .map(file => path.resolve(imgsPath, file))
    .map(filePath => cv.imread(filePath))
    .map(img => img.bgrToGray())
    .map(getFaceImage)
    .map(faceImg => faceImg.resize(80, 80));


const isImageFour = (_, i) => imgFiles[i].includes('4');
const isNotImageFour = (_, i) => !isImageFour(_, i);

const trainImages = images.filter(isNotImageFour);
const testImages = images.filter(isImageFour);


const labels = imgFiles
    .filter(isNotImageFour)
    .map(file => nameMappings.findIndex(name => file.includes(name)));


const runPrediction = (recognizer) => {
  testImages.forEach((img) => {
    const result = recognizer.predict(img);
    console.log('predicted: %s, confidence: %s', nameMappings[result.label], result.confidence);
  });
};

// const eigen = new cv.EigenFaceRecognizer();
// const fisher = new cv.FisherFaceRecognizer();
const lbph = new cv.LBPHFaceRecognizer();

eigen.train(trainImages, labels);
fisher.train(trainImages, labels);
lbph.train(trainImages, labels);

runPrediction(lbph);