## Prerequisite
For running GPU models, Docker should be able to access GPUs. For that, you need to install:
    - Linux: [nvidia-container-runtime](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu)
    - Windows: [Docker Desktop WSL 2 backend](https://docs.docker.com/desktop/windows/wsl/#turn-on-docker-desktop-wsl-2)

But you can still build, push, and pull GPU models without it.

## Declare as a GPU model

You can set ``gpu=True`` in the ``tungstenkit.model_config`` decorator:

```python hl_lines="23"
import json
from pathlib import Path
from typing import List

import torch
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, MobileNetV2

from tungstenkit import BaseIO, Field, Image, TungstenModel, model_config

LABELS = json.loads(Path("imagenet_labels.json").read_text())


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    score: float
    label: str = Field(choices=LABELS)


@model_config(
    gpu=True,
    description="Image classification model",
    python_packages=["torch", "torchvision"],
    batch_size=16,
)
class Model(TungstenModel[Input, Output]):
    def setup(self):
        """Load the model into memory"""

        self.model = MobileNetV2()
        weights = torch.load("mobilenetv2_weights.pth")
        self.model.load_state_dict(weights)
        self.model.eval()

    def predict(self, inputs: List[Input]) -> List[Output]:
        """Run a batch prediction"""

        print("Preprocessing")
        transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
        pil_images = [inp.image.to_pil_image() for inp in inputs]
        tensors = [transform(img) for img in pil_images]
        input_tensor = torch.stack(tensors, dim=0)

        print("Inferencing")
        softmax = self.model(input_tensor).softmax(1)

        print("Postprocessing")
        scores, class_indices = torch.max(softmax, 1)
        pred_labels = [LABELS[idx.item()] for idx in class_indices]
        return [
            Output(score=score.item(), label=label) for score, label in zip(scores, pred_labels)
        ]
```
Then, Tungstenkit automatically selects a compatible CUDA version and installs it in the container.
The CUDA version inference is currently supported on ``torch``, ``torchvision``, ``torchaudio``, and ``tensorflow``.

## Manually set the CUDA version
You can also pass ``cuda_version`` as an argument of the ``tungstenkit.model_config`` decorator:

```python hl_lines="23-24"
import json
from pathlib import Path
from typing import List

import torch
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, MobileNetV2

from tungstenkit import BaseIO, Field, Image, TungstenModel, model_config

LABELS = json.loads(Path("imagenet_labels.json").read_text())


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    score: float
    label: str = Field(choices=LABELS)


@model_config(
    gpu=True,
    cuda_version="11.6",
    description="Image classification model",
    python_packages=["torch", "torchvision"],
    batch_size=16,
)
class Model(TungstenModel[Input, Output]):
    def setup(self):
        """Load the model into memory"""

        self.model = MobileNetV2()
        weights = torch.load("mobilenetv2_weights.pth")
        self.model.load_state_dict(weights)
        self.model.eval()

    def predict(self, inputs: List[Input]) -> List[Output]:
        """Run a batch prediction"""

        print("Preprocessing")
        transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
        pil_images = [inp.image.to_pil_image() for inp in inputs]
        tensors = [transform(img) for img in pil_images]
        input_tensor = torch.stack(tensors, dim=0)

        print("Inferencing")
        softmax = self.model(input_tensor).softmax(1)

        print("Postprocessing")
        scores, class_indices = torch.max(softmax, 1)
        pred_labels = [LABELS[idx.item()] for idx in class_indices]
        return [
            Output(score=score.item(), label=label) for score, label in zip(scores, pred_labels)
        ]
```

