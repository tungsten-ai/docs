A Tungsten model is defined as a class with ``define_model`` decorator in ``tungsten_model.py``:

```python
from typing import List
import torch
from tungstenkit import BaseIO, Image, define_model


class Input(BaseIO):
    prompt: str


class Output(BaseIO):
    image: Image


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=4,
    gpu_mem_gb=16,
)
class TextToImageModel:
    def setup(self):
        weights = torch.load("./weights.pth")
        self.model = load_torch_model(weights)

    def predict(self, inputs: List[Input]) -> List[Output]:
        input_tensor = preprocess(inputs)
        output_tensor = self.model(input_tensor)
        outputs = postprocess(output_tensor)
        return outputs

```

Here you can find the elements needed to define a Tungsten model:

  - Input/Output classes
  - ``setup`` method
  - ``predict`` method
  - Runtime configuration (e.g. batch size & device type)
  - Dependencies (e.g. Python packages)


## Basic Usage
### Declare input/output types
Input/output types are declared by passing them as arguments of ``define_model`` decorator:

```python hl_lines="15-16"
from typing import List
import torch
from tungstenkit import BaseIO, Image, define_model


class Input(BaseIO):
    prompt: str


class Output(BaseIO):
    image: Image


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=4,
    gpu_mem_gb=16,
)
class TextToImageModel:
    def setup(self):
        weights = torch.load("./weights.pth")
        self.model = load_torch_model(weights)

    def predict(self, inputs: List[Input]) -> List[Output]:
        input_tensor = preprocess(inputs)
        output_tensor = self.model(input_tensor)
        outputs = postprocess(output_tensor)
        return outputs
```

See [Tungsten Model - Input/Output](https://tungsten-ai.github.io/docs/tungsten_model/input_and_output) to learn how to define input/output.

### Define how to load a model
You can define the ``setup`` method for loading a model:

```python hl_lines="23-25"
from typing import List
import torch
from tungstenkit import BaseIO, Image, define_model


class Input(BaseIO):
    prompt: str


class Output(BaseIO):
    image: Image


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=4,
    gpu_mem_gb=16,
)
class TextToImageModel:
    def setup(self):
        weights = torch.load("./weights.pth")
        self.model = load_torch_model(weights)

    def predict(self, inputs: List[Input]) -> List[Output]:
        input_tensor = preprocess(inputs)
        output_tensor = self.model(input_tensor)
        outputs = postprocess(output_tensor)
        return outputs
```

As you can see, the ``weights.pth`` file is required to setup.
Before building, you should make sure that the file exists in the build directory.

### Define how a prediction works
The ``predict`` method defines the computation performed at every prediction request.

It takes a non-empty list of ``Input`` objects as an argument, and should return a list of the same number of ``Output`` objects.

```python hl_lines="27-33"
from typing import List
import torch
from tungstenkit import BaseIO, Image, define_model


class Input(BaseIO):
    prompt: str


class Output(BaseIO):
    image: Image


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=4,
    gpu_mem_gb=16,
)
class TextToImageModel:
    def setup(self):
        weights = torch.load("./weights.pth")
        self.model = load_torch_model(weights)

    def predict(self, inputs: List[Input]) -> List[Output]:
        input_tensor = preprocess(inputs)
        output_tensor = self.model(input_tensor)
        outputs = postprocess(output_tensor)
        return outputs
```

During runtime, Tungstenkit automatically keeps all options same in a batch.
So, you can define the ``predict`` method of a text-to-image generaton model like this:

```python hl_lines="6-13 21 23"
from typing import List
from tungstenkit import BaseIO, Image, define_model, Option

class Input(BaseIO):
    prompt: str
    width: int = Option(
        choices=[128, 256],
        default=256,
    )
    height: int = Option(
        choices=[128, 256],
        default=256,
    )

class Output(BaseIO):
    image: Image

@define_model(input=Input, output=Output)
class TextToImageModel:
    def predict(self, inputs: List[Input]) -> List[Output]:
        options = inputs[0]
        prompts = [inp.prompt for inp in inputs]
        pil_images = self.model(prompts, width=options.width, height=options.height)
        images = [Image.from_pil_image(pil_image) for pil_image in pil_images]
        outputs = [Output(image=image) for image in images]
        return outputs
```

### Declaring dependencies and runtime configuration
You can declare dependencies and runtime configuration via ``define_model`` decorator:
```python hl_lines="17-20"
from typing import List
import torch
from tungstenkit import BaseIO, Image, define_model


class Input(BaseIO):
    prompt: str


class Output(BaseIO):
    image: Image


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    python_packages=["torch", "torchvision"],
    batch_size=4,
    gpu_mem_gb=16,
)
class TextToImageModel:
    def setup(self):
        weights = torch.load("./weights.pth")
        self.model = load_torch_model(weights)

    def predict(self, inputs: List[Input]) -> List[Output]:
        input_tensor = preprocess(inputs)
        output_tensor = self.model(input_tensor)
        outputs = postprocess(output_tensor)
        return outputs
```

The ``define_model`` decorator takes the following keyword-only arguments:

- ``input (BaseIO)``: The input class.
- ``output (BaseIO)``: The output class.
- ``demo_output (BaseIO | None)``: The demo output class. It is required only when the ``predict_demo``
method is defined (default: ``None``).
- ``batch_size (int)``: Max batch size for adaptive batching (default: ``1``).
- ``gpu (bool)``: Indicates if the model requires GPUs (default: ``False``).
- ``cuda_version (str | None)``: CUDA version in ``<major>[.<minor>[.<patch>]]`` format. If ``None`` (default), the cuda version will be automatically determined as compatible with ``python_packages``. Otherwise, fix the CUDA version as ``cuda_version``.
- ``gpu_mem_gb (int)``: Minimum GPU memory size required to run the model (default: ``16``). This argument will be ignored if ``gpu==False``.
- ``python_packages (list[str] | None)``: A list of pip requirements in ``<name>[==<version>]`` format. If ``None`` (default), no extra Python packages are added.
- ``python_version (str | None)``: Python version to use in ``<major>[.<minor>]`` format. If ``None`` (default), the python version will be automatically determined as compatible with ``python_packages`` while prefering the current Python version. Otherwise, fix the Python version as ``python_version``.
- ``system_packages (list[str] | None)``: A list of system packages that will installed by the system package manager (e.g. ``apt``). The default value is ``None``. This argument will be ignored while using a custom base image, because Tungstenkit cannot decide which package manager to use.
- ``mem_gb (int)``: Minimum memory size required to run the model (default: ``8``).
- ``include_files (list[str] | None)``: A list of patterns as in ``.gitignore``. If ``None`` (default), all files in the working directory and its subdirectories are added, which is equivalent to ``[*]``.
- ``exclude_files (list[str] | None)``: A list of patterns as in ``.gitignore`` for matching which files to exclude. If ``None`` (default), all hidden files and Python bytecodes are ignored, which is equivalent to ``[".*/", "__pycache__/", "*.pyc", "*.pyo", "*.pyd"]``.
- ``dockerfile_commands (list[str] | None)``: A list of dockerfile commands (default: ``None``). The commands will be executed before setting up python packages.
- ``base_image (str | None)``: Base docker image in ``<repository>[:<tag>]`` format. If ``None`` (default), the base image is automatically selected with respect to pip packages, the device type, and the CUDA version. Otherwise, use it as the base image and ``system_packages`` will be ignored.

## Advanced Usage
### Define how a demo prediction works
You can define an object detection model like this:

```python
from typing import List
from tungstenkit import BaseIO, Image, define_model


class BoundingBox(BaseIO):
    xmin: int
    xmax: int
    ymin: int
    ymax: int


class Detection(BaseIO):
    label: str
    bbox: BoundingBox


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    detections: List[Detection]


@define_model(input=Input, output=Output)
class ObjectDetectionModel:
    def setup(self):
        ...

    def predict(self, inputs: List[Input]) -> List[Output]:
        ...
```

But it doesn't contain any visualization, so you'll only get raw JSONs on the demo page.

Then, you could add the visualization result to the output:

```python hl_lines="23"
from typing import List
from tungstenkit import BaseIO, Image, define_model


class BoundingBox(BaseIO):
    xmin: int
    xmax: int
    ymin: int
    ymax: int


class Detection(BaseIO):
    label: str
    bbox: BoundingBox


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    detections: List[Detection]
    visualized: Image


@define_model(input=Input, output=Output)
class ObjectDetectionModel:
    def setup(self):
        ...

    def predict(self, inputs: List[Input]) -> List[Output]:
        ...
```

This can improve the demo page, but introduces the visualization overhead of the API.

In such a case, you can separate the method for demo predictions:

```python hl_lines="25-26 41-49"
from typing import List, Tuple, Dict
from tungstenkit import BaseIO, Image, define_model


class BoundingBox(BaseIO):
    xmin: int
    xmax: int
    ymin: int
    ymax: int


class Detection(BaseIO):
    label: str
    bbox: BoundingBox


class Input(BaseIO):
    image: Image


class Output(BaseIO):
    detections: List[Detection]


class Visualization(BaseIO):
    result: Image


@define_model(
    input=Input, 
    output=Output,
    demo_output=Visualization,
)
class ObjectDetectionModel:
    def setup(self):
        ...

    def predict(self, inputs: List[Input]) -> List[Output]:
        ...
    
    def predict_demo(
        self, 
        inputs: List[Input]
    ) -> Tuple[List[Output], List[Visualization]]:
        outputs = self.predict(inputs)
        pil_images = visualize(inputs, outputs)
        images = [Image.from_pil_image(pil_image) for pil_image in pil_images]
        visualizations = [Visualization(result=image) for image in images]
        return outputs, visualizations
```
Then, the ``predict`` method is executed when a prediction is requested through the API, but the ``predict_demo`` method is called for a demo request.