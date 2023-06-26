# nsfw-detect-onnx

convert HDF5model from [https://github.com/GantMan/nsfw_model](https://github.com/GantMan/nsfw_model) to onnx via [https://github.com/onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx).

## Model Download

see [releases](https://github.com/iola1999/nsfw-detect-onnx/releases).

## Demo

### python

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
```

## Acknowledgements

+ [https://github.com/GantMan/nsfw_model](https://github.com/GantMan/nsfw_model)
+ [https://github.com/onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx)
+ Generative Pre-trained Transformer 4
