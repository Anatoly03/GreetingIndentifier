import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'
import * as use from '@tensorflow-models/universal-sentence-encoder'

import comments from './train.js'
import comment_testing from './test.js'

const outputData = tf.tensor2d(
    comments.map((comment) => [
        comment.intent === 'buy' ? 1 : 0,
        comment.intent === 'none' ? 1 : 0,
    ])
)

const encodeData = (data) => {
    const sentences = data.map((comment) => comment.text.toLowerCase())
    const trainingData = use
        .load()
        .then((model) => {
            return model.embed(sentences).then((embeddings) => {
                return embeddings
            })
        })
        .catch((err) => console.error('Fit Error:', err))

    return trainingData
}

const model = tf.sequential()

// Add layers to the model
model.add(
    tf.layers.dense({
        inputShape: [512],
        activation: 'sigmoid',
        units: 2,
    })
)

model.add(
    tf.layers.dense({
        inputShape: [2],
        activation: 'sigmoid',
        units: 2,
    })
)

model.add(
    tf.layers.dense({
        inputShape: [2],
        activation: 'sigmoid',
        units: 2,
    })
)

// Compile the model
model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(0.06), // This is a standard compile config
})

function run() {
    Promise.all([encodeData(comments), encodeData(comment_testing)])
        .then((data) => {
            const { 0: training_data, 1: testing_data } = data

            model
                .fit(training_data, outputData, { epochs: 200 })
                .then((history) => {
                    model.predict(testing_data).print()
                })
        })
        .catch((err) => console.log('Prom Err:', err))
}

// Call function
run()