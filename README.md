# nsfw-detect-onnx

convert HDF5 model by [https://github.com/GantMan/nsfw_model](https://github.com/GantMan/nsfw_model) to onnx via [https://github.com/onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx).

## Model Download

see [releases](https://github.com/iola1999/nsfw-detect-onnx/releases).

## Demo

### Node.js

```js
const ort = require("onnxruntime-node");
const sharp = require("sharp");

async function loadImageAndResize(imagePath) {
  const image = await sharp(imagePath).resize(299, 299).raw().toBuffer();

  const normalizedImage = new Float32Array(1 * 299 * 299 * 3);
  for (let i = 0; i < image.length; i++) {
    normalizedImage[i] = image[i] / 255.0;
  }

  const inputTensor = normalizedImage;

  return inputTensor;
}

async function runModel(imagePath, modelPath) {
  const startTime = new Date().getTime();
  const inputTensor = await loadImageAndResize(imagePath);
  const loadImgTime = new Date().getTime();
  console.log(`Load image time: ${loadImgTime - startTime} ms`);
  const session = await ort.InferenceSession.create(modelPath);
  const modelLoadTime = new Date().getTime();
  console.log(`Load model time: ${modelLoadTime - loadImgTime} ms`);

  const inputName = session.inputNames[0];

  const options = {
    [inputName]: new ort.Tensor("float32", inputTensor, [1, 299, 299, 3]),
  };
  const makeTensorTime = new Date().getTime();
  console.log(`Make tensor time: ${makeTensorTime - modelLoadTime} ms`);
  const feeds = {};
  feeds[inputName] = options[inputName];

  const results = await session.run(feeds);
  const runModelTime = new Date().getTime();
  console.log(`Run model time: ${runModelTime - makeTensorTime} ms`);

  const outputName = session.outputNames[0];
  const output = results[outputName];
  const categories = ["drawings", "hentai", "neutral", "porn", "sexy"];
  const sortedIndices = output.data
    .map((value, index) => index)
    .sort((a, b) => output.data[b] - output.data[a]);
  const imagePreds = {};

  imagePreds[imagePath] = {};
  for (const index of sortedIndices) {
    imagePreds[imagePath][categories[index]] = output.data[index].toString();
  }

  console.log(JSON.stringify(imagePreds, null, 2));
}

const imagePath = "./images/mnzl.jpg";
const modelPath = "./model.onnx";

runModel(imagePath, modelPath);
```

Output:
```plaintext
Load image time: 52 ms
Load model time: 369 ms
Make tensor time: 1 ms
Run model time: 94 ms
{
  "../images/mnzl.jpg": {
    "hentai": "0.6869526505470276",
    "drawings": "0.2912766933441162",
    "porn": "0.01450809370726347",
    "sexy": "0.00663922680541873",
    "neutral": "0.0006233601598069072"
  }
}
```

### Python

```py
import numpy as np
import onnxruntime as ort
from PIL import Image

image_path = './images/95d4h.jpg'

image = Image.open(image_path)
image = image.resize((299, 299))

input_data = np.expand_dims(np.array(image), axis=0).astype(np.float32)
input_data /= 255.0

sess = ort.InferenceSession("model.onnx")

input_name = sess.get_inputs()[0].name

result = sess.run(None, {input_name: input_data})

categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
sorted_indices = np.argsort(result[0], axis=1).tolist()
probs = []

for i, single_indices in enumerate(sorted_indices):
    single_probs = []
    for j, index in enumerate(single_indices):
        single_probs.append(result[0][i][index])
        sorted_indices[i][j] = categories[index]
    probs.append(single_probs)

image_preds = {}
image_preds[image_path] = {}
for _ in range(len(sorted_indices[0])):
    image_preds[image_path][sorted_indices[0][_]] = str(probs[0][_])

import json
print(json.dumps(image_preds, sort_keys=True, indent=2))
# note that the sort is incorrect
```

## Acknowledgements

+ [https://github.com/GantMan/nsfw_model](https://github.com/GantMan/nsfw_model)
+ [https://github.com/onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)
+ Generative Pre-trained Transformer 4
