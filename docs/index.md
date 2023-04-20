<p align="center">
  <a href="https://tungsten-ai.github.io/tungsten-docs"><img src="images/logo.svg" alt="Tungsten" width="50%" height="50%"></a>
</p>
<p align="center">
  <img src="images/demo.gif" alt="Tungsten" width="100%" height="100%">
</p>

---

Tungsten is the easiest way to share and manage ML models. The main benefits you can enjoy are:

🚀 **Hessle-free model sharing**  
Have you ever spent a lot of time setting up CUDA and the conda environment to use a model?
Also, have you ever felt lazy about matching versions of codes and weights?
We have, yet Tungsten enables to put all together and build a standardized Docker container *by just writing a few lines of Python codes*.
So, others can run it without any model-specific setup.
Also, Tungsten wraps it with a web UI as easy as non-developers can run.
The web UI is available on both the Tungsten platform and a local machine.

⚙️ **Systematic model management**  
If models, data and the test spec can change over time, the situation becomes chaotic easily. 
Whenever test data or the test spec is updated, all evaluation scores become outdated, so you should evaluate all models again. 
Also, each model has its own implementation to calculate it, but you may not trust all of them. 
Using Tungsten, you can keep all evaluation scores up-to-date.
Tungsten detects updates and automatically runs evaluations.

## Features
- [Require only a few lines of Python codes to containerize a model](#require-only-a-few-lines-of-python-codes-to-containerize-a-model)
- Automatically generate a RESTful API for a model
- Provide a clean and intuitive web UI for a model
- Model, test data, and test spec versioning
- Keep test scores up-to-date
- Allow your own machines to be used to run remote predictions

---

## Take the tour
### Require only a few lines of Python codes to containerize a model
You don't have to write a DockerFile or any complex configuration file.

Define a Tungsten model in ``tungsten_model.py``:
```python
from typing import List, Tuple

import torch
from tungstenkit.io import BaseIO, Image
from tungstenkit.model import TungstenModel, config


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    score: float
    label: str


@config(
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=64,
    description="A torch model"
)
class Model(TungstenModel[Input, Output]):
    def setup(self):
        self.model = torch.load("./weights.pth")

    def predict(self, inputs: List[Input]) -> List[Output]:
        input_tensor = preprocess(inputs)
        output_tensor = self.model(input_tensor)
        outputs = postprocess(output_tensor)
        return outputs
```

Containerize the model:
```
tungsten build
```
That's it!

## How it works
### Define a model
Define a Tungsten model in ``tungsten_model.py``:
```python
from typing import List, Tuple

import torch
from tungstenkit.io import BaseIO, Image
from tungstenkit.model import TungstenModel, config


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    score: float
    label: str


@config(
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=64,
    description="A torch model"
)
class Model(TungstenModel[Input, Output]):
    def setup(self):
        self.model = torch.load("./weights.pth")

    def predict(self, inputs: List[Input]) -> List[Output]:
        input_tensor = preprocess(inputs)
        output_tensor = self.model(input_tensor)
        outputs = postprocess(output_tensor)
        return outputs
```

### Build it
Containerize the model:
```
tungsten build
```

### Run it locally
Now you can run predictions locally:
```shell
# Start the web demo
tungsten demo

# Start the prediction service
tungsten serve
```

### Push it
Also, you can push the model to a Tungsten server:
```shell
# Login to a Tungsten server
tungsten login https://tungsten.example.com

# Push a model
tungsten push exampleuser/exampleproject
```

### Run it remotely
Now you can run the model remotely in the web.

## Requirements
- Python >= 3.7
- [Docker](https://docs.docker.com/engine/install/)
- (Optional) [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for running GPU models locally. You can build and push a GPU model without a GPU and nvidia-docker.

## Installation
```
pip install tungstenkit
```

## License
This project is licensed under the terms of the Apache License 2.0.
