## Installation

The first step is to install Tungstenkit.

The prerequisites for installing tungstenkit are:

- Python >= 3.7
- [Docker](https://docs.docker.com/engine/install/)

If they are ready, you can install Tungstenkit as follows:

```shell
pip install tungstenkit
```
## Run an example model
### Create a directory
Let's start by creating a working directory:
```shell
mkdir tungsten-quickstart
cd tungsten-quickstart
```

### Build a model

To build a Tungsten model, you should define your input, output, setup & predict functions, and dependencies in ``tungsten_model.py`` file.

For example, you can define them for an image classification model like this:
```python
from typing import List

import torch
from torchvision import transforms
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, MobileNetV2
from tungstenkit.io import BaseIO, Image
from tungstenkit.model import TungstenModel, config


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    score: float
    label: str


@config(
    description="Image classification model",
    python_packages=["torch", "torchvision"],
    batch_size=64,
)
class Model(TungstenModel[Input, Output]):
    def setup(self):
        self.model = MobileNetV2()
        self.model.load_state_dict(torch.load("mobilenetv2_weights.pth"))
        self.model.eval()

        self.labels = MobileNet_V2_Weights.IMAGENET1K_V2.meta["categories"]
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, inputs: List[Input]) -> List[Output]:
        input_tensor = self._preprocess(inputs)
        softmax = self.model(input_tensor).softmax(1)
        scores, class_indices = torch.max(softmax, 1)
        pred_labels = [self.labels[idx.item()] for idx in class_indices]
        return [
            Output(score=score.item(), label=label) for score, label in zip(scores, pred_labels)
        ]

    def _preprocess(self, inputs: List[Input]) -> torch.Tensor:
        pil_images = [inp.image.to_pil_image() for inp in inputs]
        tensors = [self.transforms(img) for img in pil_images]
        input_tensor = torch.stack(tensors, dim=0)
        return input_tensor
```
Copy that to a file ``tungsten_model.py``.


To setup a model, model weights should be downloaded:
```
curl -o mobilenetv2_weights.pth https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth
```

Now, you can build a model using the definition:
```console
$ tungsten build -n tungsten-example

✅ Successfully built tungsten model: 'tungsten-example:latest'
```

You can see that the model you've just created is added to the model list:
```shell
tungsten models
```


### Run locally
Now, you can test the model in your local machine by running predictions.

Tungstenkit provides multiple options for that.

#### Option 1: an interactive web demo
```
tungsten demo -p 8080
```
Visit [http://localhost:8080](http://localhost:8080) to check.

#### Option 2: a RESTful API
```
tungsten serve -p 3000
```
Visit [http://localhost:3000/docs](http://localhost:3000/docs) to check.

### Run remotely
To do this, you should have an account and an entered project in a Tungsten server running at [https://tungsten-ai.com](https://tungsten-ai.com).  

If you have them, let's login first.
```shell
tungsten login
```

Then, you can push the built model:
```shell
tungsten push <username>/<project name>
```

Now you can find a new model is added to the project.

Visit [https://tungsten-ai.com](https://tungsten-ai.com) in a browser and run it.

Also, you can pull the model as follows:
```
tungsten pull <username>/<project name>:<model version>
```
## Upgrade the example
<!-- ## Use GPUs
To run GPU models locally, [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) should be installed. -->