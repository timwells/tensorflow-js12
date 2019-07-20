// Credit to: LearnCode.academy
// https://www.youtube.com/watch?v=XdErOpUzupY
//
// https://help.github.com/en/articles/adding-an-existing-project-to-github-using-the-command-line
////////////////////////////////////////////////

const tf = require('@tensorflow/tfjs');
// https://stackoverflow.com/questions/52355693/tensorflow-js-save-model-using-node
require('@tensorflow/tfjs-node')

const iris = require('./iris.json');
const irisTesting = require('./iris-testing.json');

// convert/setup our data
const trainingData = tf.tensor2d(iris.map(item => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
]))
const outputData = tf.tensor2d(iris.map(item => [
  item.species === "setosa" ? 1 : 0,
  item.species === "virginica" ? 1 : 0,
  item.species === "versicolor" ? 1 : 0,
]))

const testingData = tf.tensor2d(irisTesting.map(item => [
  item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
]))

// build neural network

// L0                 L1  L2
// [sepal_length]     []  [setosa]
// [sepal_width]      []  [virginica]
// [petal_length]     []  [versicolor]
// [petal_wdith]      []
//                    []

const model = tf.sequential()

model.add(tf.layers.dense({
  inputShape: [4],
  activation: "sigmoid",
  units: 5,
}))
model.add(tf.layers.dense({
  inputShape: [5],
  activation: "sigmoid",
  units: 3,
}))
model.add(tf.layers.dense({
  activation: "sigmoid",
  units: 3,
}))
model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(.03),
})
// train/fit our network
const startTime = Date.now()
model.fit(trainingData, outputData, {epochs: 4000})
  .then((results) => {
    // Dump last 10 loss values
    for(let i =  results.history.loss.length - 10; i < results.history.loss.length; i++) {
      console.log(`history[${i}]  ${results.history.loss[i]}`);
    }
    console.log('Training Time: ', Date.now() - startTime)
    model.predict(testingData).print()
    model.save('file:///./work/Redsnipe/TensorFlowJS/learncode/tensorflow-js12/model');
  })
// test network
