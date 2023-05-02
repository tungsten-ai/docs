<p align="center">
  <a href="https://tungsten-ai.github.io/docs"><img src="images/logo.svg" alt="Tungsten" width="50%" height="50%"></a>
</p>
<p align="center">
<a href="https://pypi.org/project/tungstenkit" target="_blank">
    <img src="https://img.shields.io/pypi/v/tungstenkit?color=%2334D058&label=pypi%20package" alt="Tungstenkit version">
</a>
<a href="https://pypi.org/project/tungstenkit" target="_blank">
    <img src="https://static.pepy.tech/badge/tungstenkit?style=flat-square" alt="Downloads">
</a>
<a href="https://pypi.org/project/tungstenkit" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/tungstenkit.svg?color=%2334D058" alt="Supported Python versions">
</a>
<a href="https://tungsten-ai-community.slack.com/" target="_blank">
    <img src="https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social" alt="Slack">
</a>
</p>

---
## What is Tungsten?
Tungsten is an open-source ML containerization tool/platform with a focus on developer productivity and collaboration.

Tungsten builds [a versatile and standardized container for your model](#tungsten-model).
Once built, it can run as a REST API server, GUI application, CLI application, serverless function, or scriptable Python function.

We also provide [a server application for managing and sharing your ML models](#tungsten-server).
It currently supports remote execution, test automation as well as basic versioning feature.


<p align="center">
  <img src="images/platform-model-demo.gif" alt="Tungsten Dashboard">
</p>


---

## Tungsten Model
The Tungsten model packages up everything required to run your model, and exposes a standardized API to support convenient features.


### Key Features
- **Easy**: [Requires only a few lines of Python code.](#build-a-tungsten-model)
- **Versatile**: Supports multiple usages:
    - [RESTful API server](#run-it-as-a-restful-api-server)
    - [GUI application](#run-it-as-a-gui-application)
    - [Serverless function](#run-it-as-a-serverless-function)
    - CLI application (coming soon)
    - Python function (coming soon)
- **Abstracted**: [User-defined JSON input/output.](#run-it-as-a-restful-api-server)
- **Standardized**: [Supports advanced workflows.](#run-it-as-a-restful-api-server)
- **Scalable**: Supports adaptive batching and clustering (coming soon).

See [Tungsten Model - Getting Started](https://tungsten-ai.github.io/docs/tungsten_model/getting_started/) to learn more.

---

### Take the tour
#### Build a Tungsten model
Building a Tungsten model is easy. All you have to do is write a simple ``tungsten_model.py`` like below:

```python
from typing import List

import torch
from tungstenkit import io, model


class Input(io.BaseIO):
    prompt: str


class Output(io.BaseIO):
    image: io.Image


@model.config(
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=4,
    description="Text to image"
)
class Model(model.TungstenModel[Input, Output]):
    def setup(self):
        """Load model weights"""
        weights = torch.load("./weights.pth")
        self.model = load_torch_model(weights)

    def predict(self, inputs: List[Input]) -> List[Output]:
        """Run a batch prediction"""
        input_tensor = preprocess(inputs)
        output_tensor = self.model(input_tensor)
        outputs = postprocess(output_tensor)
        return outputs
```

Now, you can start a build process with the following command:
```console
$ tungsten build

✅ Successfully built tungsten model: 'text-to-image:latest'
```


#### Run it as a RESTful API server

You can start a prediction with a REST API call.

Start a server:

```console
$ docker run -p 3000:3000 --gpus all text-to-image:latest

INFO:     Setting up the model
INFO:     Getting inputs from the input queue
INFO:     Starting the prediction service
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

Send a prediction request with a JSON payload:

```console
$ curl -X 'POST' 'http://localhost:3000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[{"prompt": "a professional photograph of an astronaut riding a horse"}]'

{
    "outputs": [{"image": "data:image/png;base64,..."}],
}
```

#### Run it as a GUI application
If you need a more user-friendly way to make predictions, start a GUI app with the following command:

```console
$ tungsten demo text-to-image:latest -p 8080

INFO:     Uvicorn running on http://localhost:8080 (Press CTRL+C to quit)
```

![tungsten-dashboard](images/local-model-demo.gif "Tungsten Dashboard")

#### Run it as a serverless function
We support remote, serverless executions via a [Tungsten server](#tungsten-server).

Push a model:

```console
$ tungsten push exampleuser/exampleproject -n text-to-image:latest

✅ Successfully pushed to 'https://server.tungsten-ai.com'
```

Now, you can start a remote prediction in the Tungsten server:

![tungsten-platform-model-demo](images/platform-model-demo.gif "Tungsten Platform Model Demo")


---

## Tungsten Server
The Tungsten server provides a platform where you can store, run, and test models.

### Key Features
- [Function-as-a-Service (FaaS)](#function-as-a-service-faas)
- [Scale with your own GPU/CPU devices](#scale-with-your-own-gpucpu-devices)
- [Project management](#project-management)
- Automated testing for CI/CD (coming soon)
 
 See [Tungsten Server - Getting Started](https://tungsten-ai.github.io/docs/tungsten_server/getting_started/) to learn more.

---

### Take the tour

#### Function-as-a-Service (FaaS)
The Tungsten server supports executing models as serverless functions.

In a browser, you can test any uploaded model:

![tungsten-platform-model-demo](images/platform-model-demo.gif "Tungsten Platform Model Demo")

Also, it is possible to make a prediction through the Tungsten server's REST API:

```console
$ curl -X 'POST' \
  'https://server.tungsten-ai.com/api/v1/projects/tungsten/text-to-image/models/2910c07e/predict' \
  -H 'Authorization: ************' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"input": {"prompt": "a professional photograph of an astronaut riding a horse"}}'

{
  "id": "c88e7de9",
  "status": "running",
}

$ curl -X 'GET' \
  'https://server.tungsten-ai.com/api/v1/predictions/c88e7de9' \
  -H 'Authorization: ************' \
  -H 'accept: application/json' \
{
  "output": {
    "image": "https://server.tungsten-ai.com/api/v1/files/1/93fd2ac4/output.png"
  },
  "status": "success"
}
```


#### Scale with your own GPU/CPU devices
You can easily scale serverless infrastructure with your Tungsten runners.

Register one or more runners with the following command:

```console
$ tungsten-runner register

Enter runner mode (pipeline, prediction) [prediction]: prediction
Enter URL of the tungsten server: https://server.tungsten-ai.com
Enter registration token: C6r5rp2PhfdXbJtFbBMhifgLDhagAc
Enter runner name [mydesktop]: myrunner 
Enter tags (comma separated) []: NVIDIA-A100
Enter GPU index to use []: 0
Runner 'myrunner' is registered - id: 245
Updated runner config
```

Then, start the runners to fetch jobs:

```console
$ tungsten-runner run

Runner 0   | running  2023-04-21 16:59:14.490 | INFO     | Fetching a prediction job
                      2023-04-21 16:59:49.184 | INFO     | Job 0f7c50867417456ebd1389cfb74e489f assigned
Runner 1   | running  2023-04-21 16:59:14.490 | INFO     | Fetching a prediction job
```

#### Project management
In a Tungsten server, you can organize models by grouping them into projects. 

Multiple settings are unified in a project:

- Input/output schemas
- Evaluation metrics
- Test cases
- Test data

---

## License
This project is licensed under the terms of the Apache License 2.0.
