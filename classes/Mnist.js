import fs from "fs";
import { PNG } from "pngjs";
import tf from "@tensorflow/tfjs-node";

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const IMAGE_H = 28;
const IMAGE_W = 28;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const N_CLASSES = 10;
const N_DATA  = 65000;

export default class Mnist {
    constructor(nTrain = 40000, nTest = 10000) {
        this.nTrain = nTrain;
        this.nTest = nTest;
    }

    load() {
        let pixelValues = [];
        let labelValues = [];

        let labels = fs.readFileSync('labels');
        let datasetLabels = new Uint8Array(labels)
        let data = fs.readFileSync('google.png');
        let png = PNG.sync.read(data);

        // It would be nice with a checker instead of a hard coded 60000 limit here
        for (let image = 0; image <= N_DATA - 1; image++) {
            let pixels = [];

            for (let x = 0; x <= 27; x++) {
                for (let y = 0; y <= 27; y++) {
                    pixels.push(data[(image * 28 * 28) + (x + (y * 28)) + 15]);
                }
            }
            pixelValues.push(...pixels);
        }
        labelValues = datasetLabels.slice(0, N_CLASSES * N_DATA);

        this.trainImages = pixelValues.slice(0, IMAGE_SIZE * this.nTrain);
        this.testImages = pixelValues.slice(IMAGE_SIZE * this.nTrain, IMAGE_SIZE * (this.nTrain+this.nTest))
        this.trainLabels = labelValues.slice(0, N_CLASSES * this.nTrain)
        this.testLabels = labelValues.slice(N_CLASSES * this.nTrain, N_CLASSES * (this.nTrain+this.nTest))
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
}