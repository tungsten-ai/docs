# Tungstenkit: ML container made simple
[![Version](https://img.shields.io/pypi/v/tungstenkit?color=%2334D058&label=pypi%20package)](https://pypi.org/project/tungstenkit/)
[![License](https://img.shields.io/github/license/tungsten-ai/tungstenkit)](https://raw.githubusercontent.com/tungsten-ai/tungstenkit/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/tungstenkit?style=flat-square)](https://pypi.org/project/tungstenkit/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/tungstenkit.svg?color=%2334D058)](https://pypi.org/project/tungstenkit/)

[Installation](#prerequisites) | [Features](#features) | [Examples](https://tungsten-ai.github.io/docs/getting_started) | [Documentation](https://tungsten-ai.github.io/docs)

**Tungstenkit** is ML containerization tool with a focus on developer productivity and versatility.

Running other people's models isn't an easy task. Most often, you have to go through tedious steps like resolving CUDA/Python dependencies, conforming to input/output requirements, and reading a great amount of text on how to run the model. Going through all these steps to only find out that the model isn't up to your standard, is a painful experience. 😭

Standing on the shoulder of Docker, this project aims to make sharing/using ML models less painful.
We propose a standardized way to package ML models with built-in utilities for typical use cases. For starters, a model built with Tungstenkit can automatically work as a REST API server, a GUI app, and a CLI app without mode-specific code.

With Tungstenkit, sharing and consuming ML models can be quick and enjoyable.

## Prerequisites
- Python 3.7+
- [Docker](https://docs.docker.com/get-docker/)
- (Optional) To use GPUs, install [nvidia-container-runtime](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu).

## Installation
```shell
pip install tungstenkit
```

## Features
- [Tungstenkit: ML container made simple](#tungstenkit-ml-container-made-simple)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Features](#features)
  - [Take the tour](#take-the-tour)
    - [Just a few lines of Python code](#just-a-few-lines-of-python-code)
    - [Build once, use everywhere](#build-once-use-everywhere)
      - [REST API server](#rest-api-server)
      - [GUI application](#gui-application)
      - [CLI application](#cli-application)
      - [Python API](#python-api)
    - [Framework-agnostic and lightweight](#framework-agnostic-and-lightweight)
    - [Pydantic input/output models and automatic file handling](#pydantic-inputoutput-models-and-automatic-file-handling)
    - [Batch prediction](#batch-prediction)
    - [Share with the community](#share-with-the-community)
  - [Read next](#read-next)

## Take the tour
### Just a few lines of Python code
Building a Tungsten model is easy. You just have to write a ``tungsten_model.py`` like below:

```python
from typing import List
import torch
from tungstenkit import BaseIO, Image, define_model


class Input(BaseIO):
    prompt: str


class Output(BaseIO):
    image: Image


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=4,
    gpu_mem_gb=16,
)
class TextToImageModel:
    def setup(self):
        weights = torch.load("./weights.pth")
        self.model = load_torch_model(weights)

    def predict(self, inputs: List[Input]) -> List[Output]:
        input_tensor = preprocess(inputs)
        output_tensor = self.model(input_tensor)
        outputs = postprocess(output_tensor)
        return outputs

```

Now, you can start a build process from the command line:

```console
$ tungsten build . -n text-to-image

✅ Successfully built tungsten model: 'text-to-image:e3a5de56' (also tagged as 'text-to-image:latest')
```
<!--
Check the built image:
```
$ tungsten models

Repository        Tag       Create Time          Docker Image ID
----------------  --------  -------------------  ---------------
text-to-image     latest    2023-04-26 05:23:58  830eb82f0fcd
text-to-image     e3a5de56  2023-04-26 05:23:58  830eb82f0fcd
``` -->

### Build once, use everywhere

#### REST API server

First, a Tungsten model can act as a REST API server.

```console
$ tungsten serve text-to-image -p 3000

INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

You can send a prediction request with a JSON payload with the schema you defined earlier.

```console
$ curl -X 'POST' 'http://localhost:3000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[{"prompt": "a professional photograph of an astronaut riding a horse"}]'

{
    "outputs": [{"image": "data:image/png;base64,..."}]
}
```

#### GUI application
If you need a more intuitive way to make predictions, start a GUI app with the following command.

```console
$ tungsten demo text-to-image -p 8080

INFO:     Uvicorn running on http://localhost:8080 (Press CTRL+C to quit)
```
![tungsten-dashboard](https://github.com/tungsten-ai/assets/blob/main/common/local-model-demo.gif?raw=true "Demo GIF")

Open a browser, and visit ```http://localhost:8080``` to see the generated UI for this model.

#### CLI application
It's also possible to create a prediction from the terminal:
```console
$ tungsten predict text-to-image \
   -i prompt="a professional photograph of an astronaut riding a horse"

{
  "image": "./output.png"
}
```

#### Python API
If you want to run the model inside your Python script, use the Python API:
```python
>>> from tungstenkit import models
>>> model = models.get("text-to-image")
>>> model.predict(
    {"prompt": "a professional photograph of an astronaut riding a horse"}
)
{"image": PosixPath("./output.png")}
```

### Framework-agnostic and lightweight
Tungstenkit does not have any dependencies on ML libraries.
Import any packages you want, but don't forget to declare the dependencies explicitly.

```python
# The latest cpu-only build of Tensorflow will be included
@model_config(gpu=False, python_packages=["tensorflow"])
class Model(TungstenModel[Input, Output]):
    def predict(self, inputs):
        """Run a batch prediction"""
        # ...ops using tensorflow...
        return outputs
```

### Pydantic input/output models and automatic file handling

We support the powerful Pydantic package for modeling and validating the input/output bodies. So you can use standard Python type hints to declare io requirements. In addition, we provide utility classes for a better file-handling experience.

```python
from tungstenkit import BaseIO, Image, define_model


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    image: Image


@define_model(input=Input, output=Output)
class StyleTransferModel:
    ...
```
Here, the `Image` type is used to declare the image field in both the request and response body. If a client sends invalid image bytes, it will return a nice and clear error, indicating that the prediction cannot be created. The mapped objects will include helper methods to be used inside the `predict` method (see below). Tungstenkit currently provides four classes for file handling - ``Image``, ``Audio``, ``Video``, and ``Binary``.

```python
class StyleTransferModel:
    def predict(self, inputs: List[Input]) -> List[Output]:
        # Preprocessing
        input_pil_images = [inp.image.to_pil_image() for inp in inputs]
        # Inference
        output_pil_images = do_inference(input_pil_images)
        # Postprocessing
        output_images = [Image.from_pil_image(pil_image) for pil_image in output_pil_images]
        outputs = [Output(image=image) for image in output_images]
        return outputs
```

### Batch prediction
Tungstenkit supports both manual and automatic batching.

- **Automatic batching**
     Automatic or "dynamic" batching allows a model server to group multiple prediction requests in a short timespan for higher throughput. You can configure the allowed batch size limit in `tungsten_model.py`:
    ```python
    @define_model(input=Input, output=Output, gpu=True, batch_size=32)
    ```

- **Manual batching**
    If batching is enabled, you can also manually create a batch by sending input bodies in a list.
    ```console
    $ curl -X 'POST' 'http://localhost:3000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '[{"field": "input1"}, {"field": "input2"}, {"field": "input3"}]'

    {
      "outputs": [
        {"field": "output1"},
        {"field": "output2"},
        {"field": "output3"}
      ]
    }
    ```

### Share with the community
If you like Tungstenkit, consider visiting ```tungsten.run```: a platform for sharing, running, and discussing open-source AI.


## Read next
- [Examples](https://tungsten-ai.github.io/docs/getting_started)
- [Advanced User Guide](https://tungsten-ai.github.io/docs/usage/use_gpus)
