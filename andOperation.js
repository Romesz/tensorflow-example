const tf = require('@tensorflow/tfjs');
const fs = require('fs');

// struct
let data = {
  inputs: [
    /*
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
    */
  ],
  targets: [
    /*
    [0],
    [0],
    [0],
    [1]
    */
  ]
};

// generate json
/*
for (let i = 0 ; i < 999999 ; i++) {
  data.inputs.push([0, 0]);
  data.targets.push([0]);

  data.inputs.push([1, 0]);
  data.targets.push([0]);
  
  data.inputs.push([0, 1]);
  data.targets.push([0]);

  data.inputs.push([1, 1]);
  data.targets.push([1]);
}
fs.writeFile('./data/and.json', JSON.stringify(data), () => console.log('ready'));
*/

// load json data
data = JSON.parse(fs.readFileSync('./data/and.json'));
const model = tf.sequential();
const inputLayer = tf.layers.dense({
  units: 2,
  inputShape: [2],
  activation: 'sigmoid'
});
const outputLayer = tf.layers.dense({
  units: 1,
  activation: 'sigmoid'
});

model.add(inputLayer);
model.add(outputLayer);
 
// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
 
// Generate some synthetic data for training.
const xs = tf.tensor2d(data.inputs);
const ys = tf.tensor2d(data.targets);
 
// Train the model using the data.
model.fit(xs, ys).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([[0, 0]])).print(); // around 0
  model.predict(tf.tensor2d([[1, 0]])).print(); // around 1
  model.predict(tf.tensor2d([[0, 1]])).print(); // around 1
  model.predict(tf.tensor2d([[1, 1]])).print(); // around 0
});