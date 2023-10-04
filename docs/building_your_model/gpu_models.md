## Declare as a GPU model

You can set ``gpu=True`` in the ``define_model`` decorator:

```python hl_lines="25"
import json
from pathlib import Path
from typing import List

import torch
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, MobileNetV2

from tungstenkit import BaseIO, Field, Image, define_model

LABELS = json.loads(Path("imagenet_labels.json").read_text())


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    score: float
    label: str = Field(choices=LABELS)


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=16,
)
class ImageClassifier:
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
Then, Tungstenkit automatically selects a compatible CUDA version and installs it in the container if required.
The CUDA version inference is currently supported on ``torch``, ``torchvision``, ``torchaudio``, and ``tensorflow``.

## Declare GPU memory size
You can define the minimum GPU memory size required to run this model. 

```python hl_lines="25-26"
import json
from pathlib import Path
from typing import List

import torch
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, MobileNetV2

from tungstenkit import BaseIO, Field, Image, define_model

LABELS = json.loads(Path("imagenet_labels.json").read_text())


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    score: float
    label: str = Field(choices=LABELS)


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    gpu_mem_gb=6,
    python_packages=["torch", "torchvision"],
    batch_size=16,
)
class ImageClassifier:
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

Then, the VM type is automatically determined when you push the model to [tungsten.run](https://tungsten.run). By default, the minimum GPU memory size is set as 16GB.

## Manually set the CUDA version
You can also pass ``cuda_version`` as an argument of the ``tungstenkit.define_model`` decorator:

```python hl_lines="25-26"
import json
from pathlib import Path
from typing import List

import torch
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, MobileNetV2

from tungstenkit import BaseIO, Field, Image, define_model

LABELS = json.loads(Path("imagenet_labels.json").read_text())


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    score: float
    label: str = Field(choices=LABELS)


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    cuda_version="11.8",
    python_packages=["torch", "torchvision"],
    batch_size=16,
)
class ImageClassifier:
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

## Force to install system CUDA
By default, Tungstenkit doesn't install system CUDA unless a known GPU package requires it. But in some cases, you may want to change this behavior. For this, you can use ``force_install_system_cuda`` flag of the ``tungstenkit.define_model`` decorator.

```python hl_lines="25-27"
import json
from pathlib import Path
from typing import List

import torch
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, MobileNetV2

from tungstenkit import BaseIO, Field, Image, define_model

LABELS = json.loads(Path("imagenet_labels.json").read_text())


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    score: float
    label: str = Field(choices=LABELS)


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    cuda_version="11.8",
    force_install_system_cuda=True,
    python_packages=["torch", "torchvision"],
    batch_size=16,
)
class ImageClassifier:
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