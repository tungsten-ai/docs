<p align="center">
  <a href="https://tungsten-docs.github.io"><img src="images/logo.svg" alt="Tungsten" width="50%" height="50%"></a>
</p>


Tungsten is the easiest way to build, share, and test ML models. The key features are:

- Require only a few lines of Python codes to containerize a model
- Automatically generate a RESTful API for a model
- Provide a clean and intuitive web UI for a model
- Model, test data, and test spec versioning
- Automatic testing
- Allow your own machines to be used to run remote predictions


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
