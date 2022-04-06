import tf from "@tensorflow/tfjs-node";
import MnistFileLoader from "./classes/MnistFileLoader.js";
let model = null;

let mfl = new MnistFileLoader(1000, 100);
mfl.readMNIST();

training();

function training() {
    let [x,y]= mfl.getTrainData();
    x=x.reshape([mfl.nTrain, 784]);
    console.log(y.arraySync());
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

    let epochs = 100;
    model.fit(x, y, {
        batchSize: 10,
        epochs: epochs,
    }).then(() => {
        model.save('file://./my-model');
        testing();
    }).catch(e => console.log(e));
}

async function load() {
    return  await tf.loadLayersModel('file://./my-model/model.json');
}

function testing() {
    let testSize = mfl.nTest;
    console.log(testSize);
    let [xtest,ytest] = mfl.getTestData(testSize);
    xtest=xtest.reshape([testSize,784]);
    let pred = model.predict(xtest);
    let predArr = pred.arraySync();
    let labelArr = ytest.arraySync();
    let correctCount = 0;
    for (let i = 0; i < predArr.length; i++) {
        const predictionVal = to(predArr[i]);
        const label = to(labelArr[i]);
        if (predictionVal === label) {
            correctCount++;
        }
        console.log(`prediction is: ${predictionVal}, label was: ${label}`);
        console.log('------');
    }
    console.log(`Predicted ${correctCount}/${testSize}`);
}

function to(arr) {
    const max = Math.max(...arr);
    return arr.indexOf(max);
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
