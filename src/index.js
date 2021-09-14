import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'
import * as use from '@tensorflow-models/universal-sentence-encoder'

import comments from './train.js'
import comment_testing from './test.js'

const outputData = tf.tensor2d(
    comments.map((comment) => [
        comment.intent == 'greet' ? 1 : 0,
        comment.intent == '' ? 1 : 0,
    ])
)

const encodeData = async (data) => {
    const sentences = data.map((comment) => comment.text.toLowerCase())
    const m = await use.load()
    return m.embed(sentences).then((embeddings) => {
        return embeddings
    })
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

async function run() {
    const [training_data, testing_data] = await Promise.all([
        encodeData(comments),
        encodeData(comment_testing),
    ])
    let history = await model.fit(training_data, outputData, { epochs: 200 })
    console.log(model.predict(testing_data))
    model.predict(testing_data).print()
}

run()
