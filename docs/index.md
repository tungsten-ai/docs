<p align="center">
  <a href="https://tungsten-docs.github.io"><img src="images/logo.svg" alt="Tungsten" width="50%" height="50%"></a>
</p>
<p align="center">
    <em>The best home for your ML models</em>
</p>


Tungsten is an open-sourced, developer-frendly tool for building, sharing, testing, and deploying ML models.

The key features are:
 
📦 **Framework-agnostic & hastle-free model containerization**  

- No dependencies on specific ML frameworks  
- No complex configuration files: require only a few lines of Python codes
- Easy to setup CUDA

🚀 **Easy to share models**  

- Automatically generate a RESTful API using [FastAPI](https://fastapi.tiangolo.com/)
- Provide a clean, intuitive, and locally available web inferface
- Automatic faas deployment: enable to run predictions remotely right after pushing a model 

⚙️ **Systematic model management (TBA)**

- Model, test data, and test spec versioning
- Automatic testing
- Automatically keep model scores up-to-date

💸 **Bring your own GPUs**

- Tungsten Runners: allow your own machines to be used to run remote predictions

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
