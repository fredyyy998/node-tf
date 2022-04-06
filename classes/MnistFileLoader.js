import pkg from 'canvas';
const {createCanvas} = pkg;
import fs from "fs";
import tf from "@tensorflow/tfjs-node";

const IMAGE_H = 28
const IMAGE_W = 28
const IMAGE_SIZE = IMAGE_H * IMAGE_W
const N_CLASSES = 10
const N_DATA  = 65000

export default class MnistFileLoader {
    dataSetImages = [];
    dataSetLabels = [];
    trainImages = [];

    constructor(nTrain = 40000, nTest  = 10000) {
        this.nTrain = nTrain;
        this.nTest = nTest;
    }

    readMNIST() {
        let dataFileBuffer = fs.readFileSync('.\\train-images.idx3-ubyte');
        let labelFileBuffer = fs.readFileSync('.\\train-labels.idx1-ubyte');

        for (let image = 0; image < N_DATA; image++) {
            let pixels = [];
            for (let y = 0; y <= 27; y++)
            {
                for (let x = 0; x <= 27; x++)
                {
                    pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16]);
                }
            }
            this.dataSetImages.push(...pixels);
            this.dataSetLabels.push(...this.getLabelArr(labelFileBuffer[image + 8]));
        }
        this.trainImages =
            this.dataSetImages.slice(0, IMAGE_SIZE * this.nTrain);
        this.testImages = this.dataSetImages.slice(IMAGE_SIZE * this.nTrain, IMAGE_SIZE * (this.nTrain+this.nTest));
        this.trainLabels = this.dataSetLabels.slice(0, N_CLASSES * this.nTrain);
        this.testLabels = this.dataSetLabels.slice(N_CLASSES * this.nTrain, N_CLASSES * (this.nTrain+this.nTest))
    }

    getTrainData() {
        const x_train = tf.tensor4d(
            this.trainImages,
            [this.trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1])
        const y_train = tf.tensor2d(
            this.trainLabels, [this.trainLabels.length / N_CLASSES, N_CLASSES])
        return [x_train, y_train]
    }

    getTestData(numExamples) {
        let x_test = tf.tensor4d(
            this.testImages,
            [this.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1])
        let y_test = tf.tensor2d(
            this.testLabels, [this.testLabels.length / N_CLASSES, N_CLASSES])

        if (numExamples != null) {
            x_test = x_test.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1])
            y_test = y_test.slice([0, 0], [numExamples, N_CLASSES])
        }
        return [x_test, y_test]
    }

    saveMNIST(start, end) {
        const canvas = createCanvas(28, 28);
        const ctx = canvas.getContext('2d');

        var pixelValues = readMNIST(start, end);

        pixelValues.forEach(function(image)
        {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (var y = 0; y <= 27; y++)
            {
                for (var x = 0; x <= 27; x++)
                {
                    var pixel = image.pixels[x + (y * 28)];
                    var colour = 255 - pixel;
                    ctx.fillStyle = `rgb(${colour}, ${colour}, ${colour})`;
                    ctx.fillRect(x, y, 1, 1);
                }
            }
            const buffer = canvas.toBuffer('image/png')
            fs.writeFileSync(`.\\images\\image${image.index}-${image.label}.png`, buffer)
        })
    }

    // this is shitty but i don't get the real way
    getLabelArr(num) {
        switch (num) {
           case 0:
               return [1,0,0,0,0,0,0,0,0,0];
           case 1:
               return [0,1,0,0,0,0,0,0,0,0];
           case 2:
               return [0,0,1,0,0,0,0,0,0,0];
           case 3:
               return [0,0,0,1,0,0,0,0,0,0];
           case 4:
               return [0,0,0,0,1,0,0,0,0,0];
           case 5:
               return [0,0,0,0,0,1,0,0,0,0];
           case 6:
               return [0,0,0,0,0,0,1,0,0,0];
           case 7:
               return [0,0,0,0,0,0,0,1,0,0];
           case 8:
               return [0,0,0,0,0,0,0,0,1,0];
           case 9:
               return [0,0,0,0,0,0,0,0,0,1];
            default:
                return [0,0,0,0,0,0,0,0,0,0];
       }
    }

}
