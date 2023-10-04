```python
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
    gpu=False,
    python_packages=["torch", "torchvision"],
    batch_size=16,
)
class ImageClassificationModel:
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