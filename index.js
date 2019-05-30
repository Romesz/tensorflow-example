const tf = require('@tensorflow/tfjs');

// struct
let data = {
  inputs: [
    // [0, 9], // [hour, minutes] 
    // [16, 11]
  ],
  targets: [
    // [36.3, 1.5], // [Latitude, Longitude] in 2 számrendszer
    // [36.3, 1.5]
  ]
};

for (let i = 0 ; i < 9999999 ; i++) {
  data.inputs.push([1, 0]);
  data.targets.push([36.3]);

  data.inputs.push([2, 0]);
  data.targets.push([50.3]);

  data.inputs.push([3, 0]);
  data.targets.push([0.3]);

  data.inputs.push([4, 0]);
  data.targets.push([45.3]);

  data.inputs.push([5, 0]);
  data.targets.push([24.3]);

  data.inputs.push([6, 0]);
  data.targets.push([78.3]);
}

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [2]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

const xs = tf.tensor2d(data.inputs);
const ys = tf.tensor2d(data.targets);

model.fit(xs, ys, {epochs: 10}).then(() => {
  model.predict(tf.tensor2d([[1, 0]])).print();
});