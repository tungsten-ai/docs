## Prerequisite
You should install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) if you want to run GPU models locally.

But you can still build, push, and pull GPU models without it.

## Declare as a GPU model

You can set ``gpu=True`` in ``model.config`` decorator:

```python hl_lines="20"
from typing import List

import torch
from torchvision import transforms
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, MobileNetV2

from tungstenkit import io, model


class Input(io.BaseIO):
    image: io.Image


class Output(io.BaseIO):
    score: float
    label: str


@model.config(
    gpu=True,
    description="Image classification model",
    python_packages=["torch", "torchvision"],
    batch_size=64,
)
class Model(model.TungstenModel[Input, Output]):
    def setup(self):
        self.model = MobileNetV2()
        self.model.load_state_dict(torch.load("mobilenetv2_weights.pth"))
        self.model.cuda()
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

    def _preprocess(self, inputs: List[Input]):
        pil_images = [inp.image.to_pil_image() for inp in inputs]
        tensors = [self.transforms(img) for img in pil_images]
        input_tensor = torch.stack(tensors, dim=0)
        input_tensor = input_tensor.cuda()
        return input_tensor
```
Then, Tungstenkit automatically selects a compatible CUDA version and installs it in the container.
The CUDA version inference is currently supported on ``torch``, ``torchvision``, ``torchaudio``, and ``tensorflow``.

## Manually set the CUDA version
You can also pass ``cuda_version`` as an argument of ``model.config``:

```python hl_lines="20-21"
from typing import List

import torch
from torchvision import transforms
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, MobileNetV2

from tungstenkit import io, model


class Input(io.BaseIO):
    image: io.Image


class Output(io.BaseIO):
    score: float
    label: str


@model.config(
    gpu=True,
    cuda_version="11.6"
    description="Image classification model",
    python_packages=["torch", "torchvision"],
    batch_size=64,
)
class Model(model.TungstenModel[Input, Output]):
    def setup(self):
        self.model = MobileNetV2()
        self.model.load_state_dict(torch.load("mobilenetv2_weights.pth"))
        self.model.cuda()
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

    def _preprocess(self, inputs: List[Input]):
        pil_images = [inp.image.to_pil_image() for inp in inputs]
        tensors = [self.transforms(img) for img in pil_images]
        input_tensor = torch.stack(tensors, dim=0)
        input_tensor = input_tensor.cuda()
        return input_tensor
```

