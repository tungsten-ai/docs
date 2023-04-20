<p align="center">
  <a href="https://tungsten-ai.github.io/tungsten-docs"><img src="images/logo.svg" alt="Tungsten" width="50%" height="50%"></a>
</p>
<p align="center">
  <img src="images/demo.gif" alt="Tungsten">
</p>

---

Tungsten is the easiest way to share and manage ML models. The main benefits you can enjoy are:

🚀 **Hessle-free model sharing**  
server, GUI application, CLI application, and FaaS function
Tungsten stores an ML model as a standardized Docker container.
So, any user can run it without any model-specific setup.
Also, Tungsten wraps it with a web UI as easy as non-developers can run.

⚙️ **Systematic model management**
all in one place  
We save every version
easily compare
easily download 
later 
If ML models, data and the test spec change over time, the situation becomes chaotic easily. 
Tungsten manages them and keep all evaluation scores up-to-date.

## Key Features
- [Require only a few lines of Python codes to containerize a model](#require-only-a-few-lines-of-python-codes-to-containerize-a-model)
- [Automatically generate a RESTful API for a model](#automatically-generate-a-restful-api-for-a-model)
- Provide a clean and intuitive web UI for a model
- Model, test data, and test spec versioning
- Keep test scores up-to-date
- Allow your own machines to be used to run remote predictions

---

## Requirements
- Python >= 3.7
- [Docker](https://docs.docker.com/engine/install/)
- (Optional) [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for running GPU models locally. You can build and push a GPU model without a GPU and nvidia-docker.

## Installation
```
pip install tungstenkit
```

---


## Take the tour
### Require only a few lines of Python codes to containerize a model
You don't have to write a DockerFile or any complex configuration file.  

Define a Tungsten model in ``tungsten_model.py`` like this:
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
tungsten build -n tungsten-example
```
That's it!

### Automatically generate a RESTful API for a model
The model container contains a standardized RESTful API. So, you can deploy using it.

Run the container:
```
docker run -p 3000 tungsten-example:latest
```
<div class="termy">

```console
$ docker run -p 3000 tungsten-example:latest

INFO:     Setting up the model
INFO:     Getting inputs from the input queue
INFO:     Starting the prediction service
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

</div>
Then, visit [http://localhost:3000](http://localhost:3000) in a browser:

![tungsten-model-api](images/model-api.png "Tungsten Model API")

### Provide a clean and intuitive web UI for a model


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



## License
This project is licensed under the terms of the Apache License 2.0.
