// Credit to: LearnCode.academy
// https://www.youtube.com/watch?v=XdErOpUzupY
//
// https://help.github.com/en/articles/adding-an-existing-project-to-github-using-the-command-line
////////////////////////////////////////////////

const tf = require('@tensorflow/tfjs');
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
  optimizer: tf.train.adam(.06),
})
// train/fit our network
const startTime = Date.now()
model.fit(trainingData, outputData, {epochs: 2000})
  .then((results) => {
    // Dummp last 10 loss values
    for(let i =  results.history.loss.length - 10; i < results.history.loss.length; i++) {
      console.log(`history[${i}]  ${results.history.loss[i]}`);
    }
    console.log('Training Time: ', Date.now() - startTime)
    model.predict(testingData).print()
  })
// test network
