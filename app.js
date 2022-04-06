import fs from "fs";
import { PNG } from "pngjs";
import tf from '@tensorflow/tfjs-node';
import Mnist from "./classes/Mnist.js";


let model = null;



let mnist = new Mnist(200, 200);
mnist.load();

training();

function training() {
    let [x,y]= mnist.getTrainData();
    x=x.reshape([mnist.nTrain, 784]);
    model = tf.sequential();
    let singleLayer = tf.layers.dense({
        units: 10,
        inputShape: [784,],
        activation: 'sigmoid',
        useBias: false
    });
    model.add(singleLayer);
    model.summary();
    model.compile({
        optimizer: tf.train.sgd(4.2),
        loss: 'meanSquaredError'
    });

    let epochs = 200;
    model.fit(x, y, {
        batchSize: 10,
        epochs: epochs,
        callbacks: {
            onEpochBegin: (epoch) => console.log(`model.fit: starting epoch ${(epoch + 1)}/${epochs}`)
        }
    }).then(() => console.log('finished')).catch(e => console.log(e));
}

// var download = function(uri, filename, callback){
//     request.head(uri, function(err, res, body){
//         console.log('content-type:', res.headers['content-type']);
//         console.log('content-length:', res.headers['content-length']);
//
//         request(uri).pipe(fs.createWriteStream(filename)).on('close', callback);
//     });
// };
//
// download('https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png', 'google.png', function(){
//     console.log('done');
// });